"""Measure per-token hidden-state L2 norms across decoder layers."""

import argparse
import gc
import os
from collections import defaultdict
from pathlib import Path

from quantum_friendly_transformer.experiments.lib.common import (
    ensure_parent,
    metadata_path,
    now_metadata,
    package_versions,
    seed_everything,
    write_json,
)
from quantum_friendly_transformer.experiments.runners.diagnostics.forward_stats import build_wikitext_records, make_batches


DEFAULT_MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "Qwen/Qwen2.5-14B",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure the L2 norm of every valid token after each decoder block."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Single sequence length; defaults to 512 when --seq-lens is omitted.",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=None,
        help="Sequence-length sweep, for example: --seq-lens 128 256 512.",
    )
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Shared WikiText token budget at every sequence length.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue with the remaining models if one checkpoint fails.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--constancy-tolerance",
        type=float,
        default=0.05,
        help="Call the norm constant only if every block median is within this relative tolerance of the first-block median.",
    )
    parser.add_argument(
        "--raw-output",
        default="experiments/outputs/raw/token_l2_norm_across_layers.parquet",
    )
    parser.add_argument(
        "--summary-output",
        default="experiments/outputs/aggregate_token_norm_across_layers/token_l2_norm_by_layer.csv",
    )
    parser.add_argument(
        "--verdict-output",
        default="experiments/outputs/aggregate_token_norm_across_layers/token_l2_norm_constancy.csv",
    )
    parser.add_argument(
        "--attention-verdict-output",
        default="experiments/outputs/aggregate_token_norm_across_layers/attention_input_l2_norm_constancy.csv",
    )
    parser.add_argument(
        "--pre-affine-verdict-output",
        default="experiments/outputs/aggregate_token_norm_across_layers/pre_affine_attention_input_l2_norm_constancy.csv",
    )
    parser.add_argument(
        "--figure-output",
        default="experiments/outputs/figures/figure_token_l2_norm_across_layers.png",
    )
    parser.add_argument(
        "--attention-figure-output",
        default="experiments/outputs/figures/figure_attention_input_l2_norm_across_layers.png",
    )
    parser.add_argument(
        "--pre-affine-figure-output",
        default="experiments/outputs/figures/figure_pre_affine_attention_input_l2_norm_across_layers.png",
    )
    return parser.parse_args()


def decoder_layers(model):
    candidates = [
        getattr(model, "layers", None),
        getattr(model, "h", None),
        getattr(model, "blocks", None),
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "decoder", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
        getattr(getattr(model, "transformer", None), "blocks", None),
        getattr(getattr(model, "gpt_neox", None), "layers", None),
    ]
    layers = next((candidate for candidate in candidates if candidate is not None), None)
    if layers is None:
        raise ValueError(f"cannot locate decoder layers for {type(model).__name__}")
    return list(layers)


def attention_input_norm(layer):
    for attribute in (
        "input_layernorm",
        "ln_1",
        "self_attn_layer_norm",
        "input_layer_norm",
    ):
        module = getattr(layer, attribute, None)
        if module is not None:
            return module
    raise ValueError(
        f"decoder layer {type(layer).__name__} has no recognized pre-attention norm"
    )


def tensor_from_output(output):
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def model_label(model_name):
    labels = {
        "meta-llama/Llama-2-7b-hf": "Llama 2 7B",
        "Qwen/Qwen2.5-14B": "Qwen2.5 14B",
        "tiiuae/Falcon-H1-34B-Base": "Falcon-H1 34B",
    }
    return labels.get(model_name, model_name.split("/")[-1])


def model_slug(model_name):
    return "".join(
        character.lower() if character.isalnum() else "-"
        for character in model_name
    ).strip("-")


def collect_model(
    torch,
    np,
    load_dataset,
    AutoModel,
    AutoTokenizer,
    args,
    model_name,
    seq_lens,
    token_budget,
):
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModel.from_pretrained(
        model_name,
        dtype=dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
        trust_remote_code=True,
    )
    model.eval()
    layers = decoder_layers(model)
    pooled = defaultdict(list)
    active_mask = None
    active_seq_len = None
    handles = []

    def record(stage, layer):
        def hook(_module, _inputs, output):
            hidden = tensor_from_output(output).detach()
            norms = torch.linalg.vector_norm(hidden.float(), dim=-1)
            values = norms[active_mask].cpu().numpy()
            pooled[(active_seq_len, stage, layer)].append(values)

        return hook

    def record_pre_affine(layer):
        def hook(module, inputs):
            hidden = inputs[0].detach().float()
            eps = getattr(
                module,
                "variance_epsilon",
                getattr(module, "eps", 1e-6),
            )
            if "rmsnorm" in type(module).__name__.lower():
                variance = hidden.pow(2).mean(dim=-1, keepdim=True)
                normalized = hidden * torch.rsqrt(variance + eps)
            else:
                centered = hidden - hidden.mean(dim=-1, keepdim=True)
                variance = centered.pow(2).mean(dim=-1, keepdim=True)
                normalized = centered * torch.rsqrt(variance + eps)
            norms = torch.linalg.vector_norm(normalized, dim=-1)
            pooled[(active_seq_len, "attention_input_pre_affine", layer)].append(
                norms[active_mask].cpu().numpy()
            )

        return hook

    handles.append(model.get_input_embeddings().register_forward_hook(record("embedding", 0)))
    for layer_index, layer in enumerate(layers, start=1):
        input_norm = attention_input_norm(layer)
        handles.append(
            input_norm.register_forward_pre_hook(record_pre_affine(layer_index))
        )
        handles.append(
            input_norm.register_forward_hook(record("attention_input", layer_index))
        )
        handles.append(layer.register_forward_hook(record("block_output", layer_index)))

    try:
        with torch.inference_mode():
            for seq_len in seq_lens:
                records = build_wikitext_records(
                    load_dataset,
                    tokenizer,
                    "wikitext-103-raw-v1",
                    "test",
                    seq_len,
                    args.max_samples,
                    token_budget,
                    args.cache_dir,
                )
                total_batches = (len(records) + args.batch_size - 1) // args.batch_size
                active_seq_len = seq_len
                for batch_index, batch in enumerate(
                    make_batches(torch, records, args.batch_size), start=1
                ):
                    input_ids = batch["input_ids"].to(args.device)
                    attention_mask = batch["attention_mask"].to(args.device)
                    active_mask = attention_mask.bool()
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    final_hidden = tensor_from_output(outputs).detach()
                    final_norms = torch.linalg.vector_norm(final_hidden.float(), dim=-1)
                    pooled[(seq_len, "final_norm", len(layers) + 1)].append(
                        final_norms[active_mask].cpu().numpy()
                    )
                    del outputs, final_hidden, final_norms, input_ids, attention_mask
                    print(
                        f"{model_name}, N={seq_len}: {batch_index}/{total_batches} batches",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()

    raw_frames = []
    summary_rows = []
    for (seq_len, stage, layer), arrays in sorted(
        pooled.items(), key=lambda item: (item[0][0], item[0][2], item[0][1])
    ):
        values = np.concatenate(arrays).astype(np.float32, copy=False)
        raw_frames.append(
            {
                "model": model_name,
                "model_label": model_label(model_name),
                "seq_len": seq_len,
                "stage": stage,
                "layer": layer,
                "values": values,
            }
        )
        p5, median, p95 = np.quantile(values, [0.05, 0.50, 0.95])
        mean = float(values.mean())
        std = float(values.std())
        summary_rows.append(
            {
                "model": model_name,
                "model_label": model_label(model_name),
                "dataset": "wikitext-103-raw-v1",
                "split": "test",
                "seq_len": seq_len,
                "hidden_dim": int(model.config.hidden_size),
                "num_layers": len(layers),
                "stage": stage,
                "layer": layer,
                "token_count": int(values.size),
                "l2_min": float(values.min()),
                "l2_p5": float(p5),
                "l2_mean": mean,
                "l2_median": float(median),
                "l2_p95": float(p95),
                "l2_max": float(values.max()),
                "l2_std": std,
                "l2_cv": std / mean if mean else float("nan"),
            }
        )

    revision = getattr(model.config, "_commit_hash", None)
    del model, tokenizer, pooled
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw_frames, summary_rows, revision


def raw_dataframe(pd, raw_frames):
    frames = []
    for item in raw_frames:
        values = item["values"]
        frames.append(
            pd.DataFrame(
                {
                    "model": item["model"],
                    "model_label": item["model_label"],
                    "seq_len": item["seq_len"],
                    "stage": item["stage"],
                    "layer": item["layer"],
                    "token_index": range(len(values)),
                    "token_l2": values,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def add_relative_medians(pd, summary):
    summary = summary.copy()
    embedding = (
        summary[summary["stage"] == "embedding"]
        .set_index(["model", "seq_len"])["l2_median"]
        .to_dict()
    )
    summary["median_ratio_to_embedding"] = summary.apply(
        lambda row: row["l2_median"]
        / embedding[(row["model"], row["seq_len"])],
        axis=1,
    )
    return summary


def constancy_table(pd, summary, tolerance):
    rows = []
    for (model_name, seq_len), group in summary.groupby(
        ["model", "seq_len"], sort=False
    ):
        embedding = float(group[group["stage"] == "embedding"].iloc[0]["l2_median"])
        trajectory = group[group["stage"] == "block_output"].sort_values("layer")
        first = float(trajectory.iloc[0]["l2_median"])
        relative = trajectory["l2_median"].astype(float) / first
        last = float(trajectory.iloc[-1]["l2_median"])
        rows.append(
            {
                "model": model_name,
                "model_label": trajectory.iloc[0]["model_label"],
                "seq_len": seq_len,
                "embedding_median_l2": embedding,
                "first_block_median_l2": first,
                "final_block_median_l2": last,
                "final_to_first_block_ratio": last / first,
                "min_to_first_block_ratio": float(relative.min()),
                "max_to_first_block_ratio": float(relative.max()),
                "max_relative_deviation": float((relative - 1.0).abs().max()),
                "tolerance": tolerance,
                "constant_within_tolerance": bool((relative - 1.0).abs().max() <= tolerance),
            }
        )
    return pd.DataFrame(rows)


def attention_constancy_table(pd, summary, tolerance):
    rows = []
    for (model_name, seq_len), group in summary.groupby(
        ["model", "seq_len"], sort=False
    ):
        trajectory = group[group["stage"] == "attention_input"].sort_values("layer")
        first = float(trajectory.iloc[0]["l2_median"])
        last = float(trajectory.iloc[-1]["l2_median"])
        relative = trajectory["l2_median"].astype(float) / first
        rows.append(
            {
                "model": model_name,
                "model_label": trajectory.iloc[0]["model_label"],
                "seq_len": seq_len,
                "first_attention_layer_median_l2": first,
                "final_attention_layer_median_l2": last,
                "final_to_first_attention_ratio": last / first,
                "min_to_first_attention_ratio": float(relative.min()),
                "max_to_first_attention_ratio": float(relative.max()),
                "max_relative_deviation": float((relative - 1.0).abs().max()),
                "tolerance": tolerance,
                "constant_within_tolerance": bool(
                    (relative - 1.0).abs().max() <= tolerance
                ),
            }
        )
    return pd.DataFrame(rows)


def pre_affine_constancy_table(pd, summary, tolerance):
    rows = []
    for (model_name, seq_len), group in summary.groupby(
        ["model", "seq_len"], sort=False
    ):
        trajectory = group[
            group["stage"] == "attention_input_pre_affine"
        ].sort_values("layer")
        expected = float(trajectory.iloc[0]["hidden_dim"]) ** 0.5
        relative = trajectory["l2_median"].astype(float) / expected
        rows.append(
            {
                "model": model_name,
                "model_label": trajectory.iloc[0]["model_label"],
                "seq_len": seq_len,
                "expected_sqrt_d": expected,
                "first_layer_median_l2": float(trajectory.iloc[0]["l2_median"]),
                "final_layer_median_l2": float(trajectory.iloc[-1]["l2_median"]),
                "min_layer_median_l2": float(trajectory["l2_median"].min()),
                "max_layer_median_l2": float(trajectory["l2_median"].max()),
                "max_relative_deviation_from_sqrt_d": float(
                    (relative - 1.0).abs().max()
                ),
                "tolerance": tolerance,
                "constant_within_tolerance": bool(
                    (relative - 1.0).abs().max() <= tolerance
                ),
            }
        )
    return pd.DataFrame(rows)


def save_figure(plt, summary, output_path, show_titles=True):
    models = summary["model"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(6.4 * len(models), 4.5),
        squeeze=False,
        sharey=False,
    )
    for ax, model_name in zip(axes[0], models):
        model_group = summary[summary["model"] == model_name]
        for seq_len, group in model_group.groupby("seq_len", sort=True):
            group = group.sort_values("layer")
            blocks = group[group["stage"].isin(["embedding", "block_output"])]
            final_norm = group[group["stage"] == "final_norm"]
            (line,) = ax.plot(
                blocks["layer"],
                blocks["l2_median"],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=f"N={seq_len}",
            )
            if not final_norm.empty:
                ax.scatter(
                    final_norm["layer"],
                    final_norm["l2_median"],
                    color=line.get_color(),
                    marker="D",
                    s=34,
                    zorder=4,
                )
        if show_titles:
            ax.set_title(model_label(model_name))
        ax.set_xlabel("Depth (0 = token embedding)")
        ax.set_ylabel(r"Token $L^2$ norm")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_path = ensure_parent(output_path)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"), dpi=220)
    plt.close(fig)

    if len(models) > 1:
        for model_name in models:
            model_output = output_path.with_name(
                f"{output_path.stem}_{model_slug(model_name)}{output_path.suffix}"
            )
            save_figure(
                plt,
                summary[summary["model"] == model_name],
                model_output,
                show_titles=False,
            )


def save_attention_figure(plt, summary, output_path, show_titles=None):
    models = summary["model"].drop_duplicates().tolist()
    if show_titles is None:
        show_titles = len(models) > 1
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(6.4 * len(models), 4.5),
        squeeze=False,
        sharey=False,
    )
    for ax, model_name in zip(axes[0], models):
        model_group = summary[
            (summary["model"] == model_name)
            & (summary["stage"] == "attention_input")
        ]
        plotted_changes = []
        for seq_len, group in model_group.groupby("seq_len", sort=True):
            group = group.sort_values("layer")
            baseline = float(group.iloc[0]["l2_median"])
            changes = 100.0 * (group["l2_median"].astype(float) / baseline - 1.0)
            plotted_changes.extend(changes.tolist())
            ax.plot(
                group["layer"],
                changes,
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=f"N={seq_len}",
            )
        if show_titles:
            ax.set_title(model_label(model_name))
        ax.set_xlabel("Attention layer")
        ax.set_ylabel(r"$L^2$ norm change (%)")
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
        lower = min(plotted_changes)
        upper = max(plotted_changes)
        span = max(upper - lower, 1.0)
        ax.set_ylim(lower - 0.2 * span, upper + 0.3 * span)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_path = ensure_parent(output_path)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"), dpi=220)
    plt.close(fig)

    if len(models) > 1:
        for model_name in models:
            model_output = output_path.with_name(
                f"{output_path.stem}_{model_slug(model_name)}{output_path.suffix}"
            )
            save_attention_figure(
                plt,
                summary[summary["model"] == model_name],
                model_output,
                show_titles=False,
            )


def save_pre_affine_figure(plt, summary, output_path, show_titles=True):
    models = summary["model"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(6.4 * len(models), 4.5),
        squeeze=False,
        sharey=False,
    )
    for ax, model_name in zip(axes[0], models):
        model_group = summary[
            (summary["model"] == model_name)
            & (summary["stage"] == "attention_input_pre_affine")
        ]
        expected = float(model_group.iloc[0]["hidden_dim"]) ** 0.5
        for seq_len, group in model_group.groupby("seq_len", sort=True):
            group = group.sort_values("layer")
            ax.plot(
                group["layer"],
                group["l2_median"],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=f"N={seq_len}",
            )
        ax.axhline(
            expected,
            color="black",
            linestyle=":",
            linewidth=1.1,
            alpha=0.7,
            label=r"theoretical $\sqrt{d}$",
        )
        if show_titles:
            ax.set_title(model_label(model_name))
        ax.set_xlabel("Attention layer")
        ax.set_ylabel(r"Pre-affine normalized token $L^2$ norm")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_path = ensure_parent(output_path)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"), dpi=220)
    plt.close(fig)

    if len(models) > 1:
        for model_name in models:
            model_output = output_path.with_name(
                f"{output_path.stem}_{model_slug(model_name)}{output_path.suffix}"
            )
            save_pre_affine_figure(
                plt,
                summary[summary["model"] == model_name],
                model_output,
                show_titles=False,
            )


def main():
    args = parse_args()
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.seq_len is not None and args.seq_lens is not None:
        raise ValueError("use either --seq-len or --seq-lens, not both")
    seq_lens = args.seq_lens or [args.seq_len or 512]
    if any(seq_len < 1 for seq_len in seq_lens):
        raise ValueError("all sequence lengths must be positive")
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    if not 0 <= args.constancy_tolerance < 1:
        raise ValueError("--constancy-tolerance must be in [0, 1)")
    token_budget = args.max_tokens or max(seq_lens) * args.max_samples
    seed_everything(args.seed)

    all_raw = []
    all_summary = []
    revisions = {}
    failures = {}
    for model_name in args.models:
        try:
            raw_frames, summary_rows, revision = collect_model(
                torch,
                np,
                load_dataset,
                AutoModel,
                AutoTokenizer,
                args,
                model_name,
                seq_lens,
                token_budget,
            )
        except Exception as error:
            if not args.skip_errors:
                raise
            failures[model_name] = f"{type(error).__name__}: {error}"
            print(f"skipping {model_name}: {failures[model_name]}", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        all_raw.extend(raw_frames)
        all_summary.extend(summary_rows)
        revisions[model_name] = revision

    if not all_summary:
        raise RuntimeError("no models completed successfully")

    raw = raw_dataframe(pd, all_raw)
    summary = add_relative_medians(pd, pd.DataFrame(all_summary))
    verdict = constancy_table(pd, summary, args.constancy_tolerance)
    attention_verdict = attention_constancy_table(
        pd,
        summary,
        args.constancy_tolerance,
    )
    pre_affine_verdict = pre_affine_constancy_table(
        pd,
        summary,
        args.constancy_tolerance,
    )

    raw_output = ensure_parent(args.raw_output)
    summary_output = ensure_parent(args.summary_output)
    verdict_output = ensure_parent(args.verdict_output)
    attention_verdict_output = ensure_parent(args.attention_verdict_output)
    pre_affine_verdict_output = ensure_parent(args.pre_affine_verdict_output)
    raw.to_parquet(raw_output, index=False)
    summary.to_csv(summary_output, index=False)
    verdict.to_csv(verdict_output, index=False)
    attention_verdict.to_csv(attention_verdict_output, index=False)
    pre_affine_verdict.to_csv(pre_affine_verdict_output, index=False)
    save_attention_figure(plt, summary, Path(args.attention_figure_output))

    write_json(
        metadata_path(raw_output),
        {
            **now_metadata(),
            "models": args.models,
            "model_revisions": revisions,
            "model_failures": failures,
            "dataset": "wikitext",
            "config": "wikitext-103-raw-v1",
            "split": "test",
            "sequence_lengths": seq_lens,
            "num_examples_per_sequence_length": args.max_samples,
            "shared_max_token_budget": token_budget,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
            "definition": (
                "Per-token L2 norm of the residual stream after the token embedding "
                "and each decoder block; attention_input is the output of each "
                "block's pre-attention LayerNorm or RMSNorm and therefore the exact "
                "tensor sent into self-attention; attention_input_pre_affine is the "
                "centered unit-variance vector for LayerNorm or the unit-RMS vector "
                "for RMSNorm, before learned affine weights; the model's final "
                "normalized output is reported separately as final_norm."
            ),
            "constancy_criterion": (
                "Every block-output median must remain within "
                f"{100 * args.constancy_tolerance:.1f}% of the first-block median."
            ),
            "attention_constancy_criterion": (
                "Every attention-input median must remain within "
                f"{100 * args.constancy_tolerance:.1f}% of the first attention layer's median."
            ),
            "pre_affine_constancy_criterion": (
                "Every pre-affine attention-input median must remain within "
                f"{100 * args.constancy_tolerance:.1f}% of sqrt(hidden_dim)."
            ),
            "raw_output": str(raw_output),
            "summary_output": str(summary_output),
            "verdict_output": str(verdict_output),
            "attention_verdict_output": str(attention_verdict_output),
            "pre_affine_verdict_output": str(pre_affine_verdict_output),
            "attention_figure_output": str(args.attention_figure_output),
            "package_versions": package_versions(),
        },
    )
    print(verdict.to_string(index=False), flush=True)
    print(attention_verdict.to_string(index=False), flush=True)
    print(pre_affine_verdict.to_string(index=False), flush=True)
    print(f"wrote {raw_output}", flush=True)
    print(f"wrote {summary_output}", flush=True)
    print(f"wrote {verdict_output}", flush=True)
    print(f"wrote {attention_verdict_output}", flush=True)
    print(f"wrote {pre_affine_verdict_output}", flush=True)
    print(f"wrote {args.attention_figure_output}", flush=True)


if __name__ == "__main__":
    main()
