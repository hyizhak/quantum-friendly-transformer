"""Measure exact attention-input token norms in parameter-shared ALBERT models."""

import argparse
import gc
from collections import defaultdict
from pathlib import Path

from quantum_friendly_transformer.experiments.lib.common import ensure_parent, metadata_path, now_metadata, write_json
from quantum_friendly_transformer.experiments.runners.diagnostics.forward_stats import build_wikitext_records, make_batches


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["albert/albert-xxlarge-v2"])
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
        default="experiments/outputs/raw/albert_shared_attention_norms.parquet",
    )
    parser.add_argument(
        "--summary-output",
        default="experiments/outputs/aggregate_albert_shared_affine/attention_input_by_layer.csv",
    )
    parser.add_argument(
        "--verdict-output",
        default="experiments/outputs/aggregate_albert_shared_affine/depth_stability.csv",
    )
    parser.add_argument(
        "--figure-output",
        default="experiments/outputs/figures/figure_albert_shared_affine_attention_input_across_layers.png",
    )
    return parser.parse_args()


def shared_attention_modules(model):
    modules = []
    for group in model.encoder.albert_layer_groups:
        for layer in group.albert_layers:
            modules.append(layer.attention)
    return modules


def collect_model(torch, np, load_dataset, AutoModel, AutoTokenizer, args, model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.float32,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    )
    model.eval()
    num_layers = int(model.config.num_hidden_layers)
    hidden_dim = int(model.config.hidden_size)
    pooled = defaultdict(list)
    active_mask = None
    active_seq_len = None
    calls = 0

    def attention_pre_hook(_module, inputs):
        nonlocal calls
        calls += 1
        hidden = inputs[0].detach().float()
        norms = torch.linalg.vector_norm(hidden, dim=-1)
        pooled[(active_seq_len, calls)].append(norms[active_mask].cpu().numpy())

    handles = [
        module.register_forward_pre_hook(attention_pre_hook)
        for module in shared_attention_modules(model)
    ]
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
                    calls = 0
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
                    if calls != num_layers:
                        raise AssertionError(
                            f"recorded {calls} attention calls for {num_layers} logical layers"
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
                "model": model_name,
                "seq_len": seq_len,
                "layer": layer,
                "token_l2": float(value),
            }
            for value in values
        )
        summary_rows.append(
            {
                "model": model_name,
                "seq_len": seq_len,
                "layer": layer,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "token_count": int(values.size),
                "l2_mean": float(values.mean()),
                "l2_median": float(np.median(values)),
                "l2_std": float(values.std()),
                "ratio_to_sqrt_d": float(np.median(values) / np.sqrt(hidden_dim)),
            }
        )

    shared_layer = model.encoder.albert_layer_groups[0].albert_layers[0]
    gamma = shared_layer.full_layer_layer_norm.weight.detach().float()
    beta = shared_layer.full_layer_layer_norm.bias.detach().float()
    affine_isotropic_prediction = float(torch.sqrt(gamma.square().sum() + beta.square().sum()))
    revision = getattr(model.config, "_commit_hash", None)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return raw_rows, summary_rows, affine_isotropic_prediction, revision


def main():
    args = parse_args()
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    raw_rows = []
    summary_rows = []
    predictions = {}
    revisions = {}
    for model_name in args.models:
        raw, summary, prediction, revision = collect_model(
            torch, np, load_dataset, AutoModel, AutoTokenizer, args, model_name
        )
        raw_rows.extend(raw)
        summary_rows.extend(summary)
        predictions[model_name] = prediction
        revisions[model_name] = revision

    raw = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summary_rows)
    verdict_rows = []
    for (model_name, seq_len), group in summary.groupby(["model", "seq_len"]):
        trajectory = group.sort_values("layer")
        values = trajectory["l2_median"].to_numpy()
        relative = values / values[0]
        verdict_rows.append(
            {
                "model": model_name,
                "seq_len": seq_len,
                "hidden_dim": int(trajectory.iloc[0]["hidden_dim"]),
                "num_layers": int(trajectory.iloc[0]["num_layers"]),
                "sqrt_d": float(np.sqrt(trajectory.iloc[0]["hidden_dim"])),
                "shared_affine_isotropic_prediction": predictions[model_name],
                "first_layer_median_l2": float(values[0]),
                "final_layer_median_l2": float(values[-1]),
                "minimum_layer_median_l2": float(values.min()),
                "maximum_layer_median_l2": float(values.max()),
                "maximum_relative_deviation_from_first": float(np.abs(relative - 1).max()),
                "maximum_over_minimum": float(values.max() / values.min()),
                "tolerance": args.tolerance,
                "constant_within_tolerance": bool(np.abs(relative - 1).max() <= args.tolerance),
            }
        )
    verdict = pd.DataFrame(verdict_rows)

    ensure_parent(args.raw_output)
    ensure_parent(args.summary_output)
    ensure_parent(args.verdict_output)
    ensure_parent(args.figure_output)
    raw.to_parquet(args.raw_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    verdict.to_csv(args.verdict_output, index=False)

    fig, ax = plt.subplots(figsize=(7.1, 5.0))
    for (model_name, seq_len), group in summary.groupby(["model", "seq_len"]):
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
            "models": args.models,
            "model_revisions": revisions,
            "sequence_lengths": args.seq_lens,
            "max_tokens": args.max_tokens,
            "definition": "Exact hidden_states argument received by each logical ALBERT self-attention call.",
            "affine_prediction": "sqrt(||gamma||_2^2 + ||beta||_2^2) for the shared block-output LayerNorm under isotropic normalized activations.",
        }
    )
    write_json(metadata_path(args.raw_output), metadata)
    print(verdict.to_string(index=False))


if __name__ == "__main__":
    main()
