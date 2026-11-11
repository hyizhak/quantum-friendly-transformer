"""Measure exact attention-input token norms in a normalized decoder-only LLM."""

import argparse
import gc
from collections import defaultdict
from pathlib import Path

from quantum_friendly_transformer.experiments.lib.common import ensure_parent, metadata_path, now_metadata, write_json
from quantum_friendly_transformer.experiments.runners.diagnostics.forward_stats import build_wikitext_records, make_batches


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="p2o6e100/nGPT_800m")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument(
        "--raw-output",
        default="experiments/outputs/raw/ngpt_800m_attention_input_norms.parquet",
    )
    parser.add_argument(
        "--summary-output",
        default="experiments/outputs/aggregate_ngpt_800m/attention_input_by_layer.csv",
    )
    parser.add_argument(
        "--verdict-output",
        default="experiments/outputs/aggregate_ngpt_800m/depth_stability.csv",
    )
    parser.add_argument(
        "--boundary-output",
        default="experiments/outputs/aggregate_ngpt_800m/attention_boundary_verification.csv",
    )
    parser.add_argument(
        "--figure-output",
        default="experiments/outputs/figures/figure_ngpt_800m_attention_input_across_layers.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
        trust_remote_code=True,
    )
    model.eval()
    layers = list(model.layers)
    hidden_dim = int(model.config.hidden_size)
    pooled = defaultdict(list)
    hook_calls = defaultdict(int)
    active_mask = None
    active_seq_len = None
    handles = []

    for layer_index, layer in enumerate(layers, start=1):
        def attention_pre_hook(_module, inputs, kwargs, index=layer_index):
            hidden = kwargs.get("hidden_states")
            if hidden is None:
                hidden = inputs[0]
            norms = torch.linalg.vector_norm(hidden.detach().float(), dim=-1)
            pooled[(active_seq_len, index)].append(norms[active_mask].cpu().numpy())
            hook_calls[(active_seq_len, index)] += 1

        handles.append(
            layer.self_attn.register_forward_pre_hook(
                attention_pre_hook,
                with_kwargs=True,
            )
        )

    try:
        with torch.inference_mode():
            for seq_len in args.seq_lens:
                records = build_wikitext_records(
                    load_dataset,
                    tokenizer,
                    "wikitext-103-raw-v1",
                    "test",
                    seq_len,
                    args.max_samples,
                    args.max_tokens,
                    args.cache_dir,
                )
                active_seq_len = seq_len
                for batch in make_batches(torch, records, args.batch_size):
                    input_ids = batch["input_ids"].to(args.device)
                    attention_mask = batch["attention_mask"].to(args.device)
                    active_mask = attention_mask.bool()
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
    finally:
        for handle in handles:
            handle.remove()

    raw_rows = []
    summary_rows = []
    for (seq_len, layer), arrays in sorted(pooled.items()):
        values = np.concatenate(arrays).astype(np.float32, copy=False)
        raw_rows.extend(
            {
                "model": args.model,
                "seq_len": seq_len,
                "layer": layer,
                "token_l2": float(value),
            }
            for value in values
        )
        summary_rows.append(
            {
                "model": args.model,
                "seq_len": seq_len,
                "layer": layer,
                "hidden_dim": hidden_dim,
                "num_layers": len(layers),
                "token_count": int(values.size),
                "l2_min": float(values.min()),
                "l2_mean": float(values.mean()),
                "l2_median": float(np.median(values)),
                "l2_max": float(values.max()),
                "l2_std": float(values.std()),
                "relative_deviation_from_unit": float(abs(np.median(values) - 1.0)),
                "ratio_to_sqrt_d": float(np.median(values) / np.sqrt(hidden_dim)),
            }
        )

    raw = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summary_rows)
    verdict_rows = []
    boundary_rows = []
    for seq_len, group in summary.groupby("seq_len"):
        trajectory = group.sort_values("layer")
        values = trajectory["l2_median"].to_numpy()
        relative = values / values[0]
        verdict_rows.append(
            {
                "model": args.model,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "num_layers": len(layers),
                "sqrt_d": float(np.sqrt(hidden_dim)),
                "first_layer_median_l2": float(values[0]),
                "final_layer_median_l2": float(values[-1]),
                "minimum_layer_median_l2": float(values.min()),
                "maximum_layer_median_l2": float(values.max()),
                "maximum_relative_deviation_from_first": float(np.abs(relative - 1).max()),
                "maximum_relative_deviation_from_unit": float(np.abs(values - 1).max()),
                "tolerance": args.tolerance,
                "constant_within_tolerance": bool(np.abs(relative - 1).max() <= args.tolerance),
            }
        )
        calls = [hook_calls[(seq_len, layer)] for layer in range(1, len(layers) + 1)]
        boundary_rows.append(
            {
                "model": args.model,
                "seq_len": seq_len,
                "layers_compared": len(calls),
                "minimum_hook_calls_per_layer": min(calls),
                "maximum_hook_calls_per_layer": max(calls),
                "all_layers_observed": bool(all(count > 0 for count in calls)),
                "measurement": "self_attn forward-pre-hook hidden_states argument",
            }
        )
    verdict = pd.DataFrame(verdict_rows)
    boundary = pd.DataFrame(boundary_rows)

    ensure_parent(args.raw_output)
    ensure_parent(args.summary_output)
    ensure_parent(args.verdict_output)
    ensure_parent(args.boundary_output)
    ensure_parent(args.figure_output)
    raw.to_parquet(args.raw_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    verdict.to_csv(args.verdict_output, index=False)
    boundary.to_csv(args.boundary_output, index=False)

    fig, ax = plt.subplots(figsize=(7.1, 5.0))
    for seq_len, group in summary.groupby("seq_len"):
        trajectory = group.sort_values("layer")
        change = 100 * (trajectory["l2_median"] / trajectory.iloc[0]["l2_median"] - 1)
        ax.plot(trajectory["layer"], change, marker="o", markersize=3, label=f"N={seq_len}")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$L^2$ norm change (%)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.figure_output, dpi=200)
    fig.savefig(str(Path(args.figure_output).with_suffix(".pdf")))
    plt.close(fig)

    metadata = now_metadata()
    metadata.update(
        {
            "model": args.model,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "sequence_lengths": args.seq_lens,
            "max_tokens": args.max_tokens,
            "definition": "Exact hidden_states argument received by each self-attention module.",
            "architecture_note": "The checkpoint uses unit-L2 normalized residual states and no pre-attention LayerNorm/RMSNorm affine map.",
        }
    )
    write_json(metadata_path(args.raw_output), metadata)
    print(verdict.to_string(index=False))
    print(boundary.to_string(index=False))

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
