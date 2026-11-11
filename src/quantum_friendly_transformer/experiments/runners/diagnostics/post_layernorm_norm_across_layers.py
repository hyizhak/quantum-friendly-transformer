"""Measure sequence-matrix norms across genuinely post-LayerNorm models."""

import argparse
import gc
import math
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
from quantum_friendly_transformer.experiments.runners.diagnostics.forward_stats import (
    build_wikitext_records,
    make_batches,
)


DEFAULT_MODELS = [
    "bert-base-uncased",
    "bert-large-uncased",
    "roberta-base",
    "FacebookAI/roberta-large",
    "google/electra-small-discriminator",
    "google/electra-base-discriminator",
    "google/electra-large-discriminator",
    "microsoft/deberta-v2-xxlarge",
    "facebook/opt-350m",
]

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure token and sequence-matrix L2 norms after the residual-plus-"
            "LayerNorm operations in BERT-style post-LayerNorm encoders."
        )
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Use the same WikiText token budget at every sequence length. "
            "Defaults to max(seq_lens) * max_samples."
        ),
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw-output",
        default="experiments/outputs/raw/post_layernorm_token_norms.parquet",
    )
    parser.add_argument(
        "--sample-output",
        default=(
            "experiments/outputs/aggregate_post_layernorm_norms/"
            "sequence_norm_by_sample.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=(
            "experiments/outputs/aggregate_post_layernorm_norms/"
            "norm_by_layer.csv"
        ),
    )
    parser.add_argument(
        "--scaling-output",
        default=(
            "experiments/outputs/aggregate_post_layernorm_norms/"
            "sequence_length_scaling_by_layer.csv"
        ),
    )
    parser.add_argument(
        "--verdict-output",
        default=(
            "experiments/outputs/aggregate_post_layernorm_norms/"
            "depth_stability.csv"
        ),
    )
    parser.add_argument(
        "--figure-output",
        default=(
            "experiments/outputs/figures/"
            "figure_post_layernorm_sequence_factor_across_layers.png"
        ),
    )
    return parser.parse_args()


def model_label(model_name):
    labels = {
        "bert-base-uncased": "BERT base (12 layers)",
        "bert-large-uncased": "BERT large (24 layers)",
        "roberta-base": "RoBERTa base (12 layers)",
        "FacebookAI/roberta-large": "RoBERTa large (24 layers)",
        "google/electra-small-discriminator": "ELECTRA small (12 layers)",
        "google/electra-base-discriminator": "ELECTRA base (12 layers)",
        "google/electra-large-discriminator": "ELECTRA large (24 layers)",
        "microsoft/deberta-v2-xxlarge": "DeBERTa-v2 XXLarge (48 layers)",
        "facebook/opt-350m": "OPT 350M post-LN (24 layers)",
    }
    return labels.get(model_name, model_name.split("/")[-1])


def post_layernorm_components(model):
    if getattr(model.config, "model_type", None) == "opt":
        if model.config.do_layer_norm_before:
            raise ValueError("OPT checkpoint is pre-LayerNorm, not post-LayerNorm")
        decoder = getattr(model, "decoder", None)
        if decoder is None:
            decoder = getattr(getattr(model, "model", None), "decoder", None)
        if decoder is None or not hasattr(decoder, "layers"):
            raise ValueError(f"cannot locate OPT decoder layers on {type(model).__name__}")
        return None, list(decoder.layers), "opt"

    base = model
    for attribute in ("bert", "roberta", "electra", "deberta"):
        candidate = getattr(model, attribute, None)
        if candidate is not None:
            base = candidate
            break
    if not hasattr(base, "encoder") or not hasattr(base, "embeddings"):
        raise ValueError(
            f"{type(model).__name__} is not a supported post-LayerNorm encoder"
        )
    return base.embeddings.LayerNorm, list(base.encoder.layer), "bert_like"


def power_spectral_norm(torch, matrix, iterations=8):
    """Deterministic power-iteration estimate used only as a spectral cross-check."""
    if matrix.numel() == 0:
        return float("nan")
    matrix = matrix.float()
    v = torch.linspace(1.0, 2.0, matrix.shape[1], device=matrix.device)
    v = v / torch.linalg.vector_norm(v)
    for _ in range(iterations):
        u = matrix @ v
        u = u / (torch.linalg.vector_norm(u) + 1e-12)
        v = matrix.transpose(0, 1) @ u
        v = v / (torch.linalg.vector_norm(v) + 1e-12)
    return float(torch.linalg.vector_norm(matrix @ v).item())


def collect_model(torch, np, load_dataset, AutoModel, AutoTokenizer, args, model_name):
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
    )
    model.eval()
    embedding_norm, layers, norm_layout = post_layernorm_components(model)
    hidden_dim = int(model.config.hidden_size)

    token_values = defaultdict(list)
    denom_values = defaultdict(list)
    sample_rows = []
    handles = []
    active_mask = None
    active_seq_len = None
    active_sample_indices = None

    def output_hook(stage, layer):
        def hook(_module, _inputs, output):
            hidden = output.detach().float()
            norms = torch.linalg.vector_norm(hidden, dim=-1)
            token_values[(active_seq_len, stage, layer)].append(
                norms[active_mask].cpu().numpy()
            )
            for batch_index, sample_index in enumerate(active_sample_indices):
                matrix = hidden[batch_index][active_mask[batch_index]]
                token_count = int(matrix.shape[0])
                frobenius = float(torch.linalg.vector_norm(matrix).item())
                spectral = power_spectral_norm(torch, matrix)
                factor = frobenius / math.sqrt(token_count)
                sample_rows.append(
                    {
                        "model": model_name,
                        "model_label": model_label(model_name),
                        "seq_len": active_seq_len,
                        "sample_index": sample_index,
                        "stage": stage,
                        "layer": layer,
                        "hidden_dim": hidden_dim,
                        "num_layers": len(layers),
                        "token_count": token_count,
                        "sequence_frobenius": frobenius,
                        "sequence_spectral_power": spectral,
                        "frobenius_over_sqrt_n": factor,
                        "frobenius_over_sqrt_nd": factor / math.sqrt(hidden_dim),
                        "spectral_over_sqrt_n": spectral / math.sqrt(token_count),
                    }
                )

        return hook

    def input_hook(stage, layer):
        record = output_hook(stage, layer)

        def hook(module, inputs):
            record(module, None, inputs[0])

        return hook

    def denominator_hook(stage, layer):
        def hook(module, inputs):
            residual_sum = inputs[0].detach().float()
            eps = float(getattr(module, "eps", 1e-12))
            centered = residual_sum - residual_sum.mean(dim=-1, keepdim=True)
            denominator = torch.sqrt(centered.pow(2).mean(dim=-1) + eps)
            denom_values[(active_seq_len, stage, layer)].append(
                denominator[active_mask].cpu().numpy()
            )

        return hook

    if embedding_norm is not None:
        handles.append(
            embedding_norm.register_forward_pre_hook(
                denominator_hook("embedding_output", 0)
            )
        )
        handles.append(
            embedding_norm.register_forward_hook(output_hook("embedding_output", 0))
        )
    for layer_index, layer in enumerate(layers, start=1):
        if norm_layout == "opt":
            post_attention = layer.self_attn_layer_norm
            post_ffn = layer.final_layer_norm
        else:
            post_attention = layer.attention.output.LayerNorm
            post_ffn = layer.output.LayerNorm
        handles.append(
            layer.register_forward_pre_hook(
                input_hook("attention_input", layer_index)
            )
        )
        handles.append(
            post_attention.register_forward_pre_hook(
                denominator_hook("post_attention_layernorm", layer_index)
            )
        )
        handles.append(
            post_attention.register_forward_hook(
                output_hook("post_attention_layernorm", layer_index)
            )
        )
        handles.append(
            post_ffn.register_forward_pre_hook(
                denominator_hook("block_output", layer_index)
            )
        )
        handles.append(
            post_ffn.register_forward_hook(output_hook("block_output", layer_index))
        )

    try:
        with torch.inference_mode():
            token_budget = args.max_tokens or max(args.seq_lens) * args.max_samples
            for seq_len in args.seq_lens:
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
                for batch_index, batch in enumerate(
                    make_batches(torch, records, args.batch_size), start=1
                ):
                    input_ids = batch["input_ids"].to(args.device)
                    attention_mask = batch["attention_mask"].to(args.device)
                    active_mask = attention_mask.bool()
                    active_seq_len = seq_len
                    active_sample_indices = batch["sample_index"]
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
                    print(
                        f"{model_name}, N={seq_len}: {batch_index}/{total_batches} batches",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()

    raw_frames = []
    summary_rows = []
    all_keys = sorted(
        set(token_values) | set(denom_values),
        key=lambda key: (key[0], key[2], key[1]),
    )
    for seq_len, stage, layer in all_keys:
        tokens = np.concatenate(token_values[(seq_len, stage, layer)]).astype(
            np.float32, copy=False
        )
        denominator_arrays = denom_values[(seq_len, stage, layer)]
        if denominator_arrays:
            denominators = np.concatenate(denominator_arrays).astype(
                np.float32, copy=False
            )
            denominator_stats = {
                "denominator_min": float(denominators.min()),
                "denominator_mean": float(denominators.mean()),
                "denominator_median": float(np.median(denominators)),
                "denominator_p95": float(np.quantile(denominators, 0.95)),
                "denominator_max": float(denominators.max()),
            }
        else:
            denominators = np.full(tokens.shape, np.nan, dtype=np.float32)
            denominator_stats = {
                "denominator_min": float("nan"),
                "denominator_mean": float("nan"),
                "denominator_median": float("nan"),
                "denominator_p95": float("nan"),
                "denominator_max": float("nan"),
            }
        raw_frames.append(
            {
                "model": model_name,
                "model_label": model_label(model_name),
                "seq_len": seq_len,
                "stage": stage,
                "layer": layer,
                "token_l2": tokens,
                "layernorm_denominator": denominators,
            }
        )
        summary_rows.append(
            {
                "model": model_name,
                "model_label": model_label(model_name),
                "seq_len": seq_len,
                "stage": stage,
                "layer": layer,
                "hidden_dim": hidden_dim,
                "num_layers": len(layers),
                "token_count": int(tokens.size),
                "token_l2_min": float(tokens.min()),
                "token_l2_mean": float(tokens.mean()),
                "token_l2_median": float(np.median(tokens)),
                "token_l2_p95": float(np.quantile(tokens, 0.95)),
                "token_l2_max": float(tokens.max()),
                **denominator_stats,
            }
        )

    revision = getattr(model.config, "_commit_hash", None)
    del model, tokenizer, token_values, denom_values
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw_frames, sample_rows, summary_rows, revision


def raw_dataframe(pd, raw_frames):
    frames = []
    for item in raw_frames:
        count = len(item["token_l2"])
        frames.append(
            pd.DataFrame(
                {
                    "model": item["model"],
                    "model_label": item["model_label"],
                    "seq_len": item["seq_len"],
                    "stage": item["stage"],
                    "layer": item["layer"],
                    "token_index": range(count),
                    "token_l2": item["token_l2"],
                    "layernorm_denominator": item["layernorm_denominator"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def aggregate_samples(pd, samples):
    numeric = [
        "sequence_frobenius",
        "sequence_spectral_power",
        "frobenius_over_sqrt_n",
        "frobenius_over_sqrt_nd",
        "spectral_over_sqrt_n",
    ]
    grouped = (
        samples.groupby(
            [
                "model",
                "model_label",
                "seq_len",
                "stage",
                "layer",
                "hidden_dim",
                "num_layers",
            ],
            as_index=False,
            sort=False,
        )[numeric]
        .agg(["mean", "std", "min", "max"])
    )
    grouped.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in grouped.columns
    ]
    return grouped


def sequence_scaling(pd, np, aggregate):
    rows = []
    for (model_name, stage, layer), group in aggregate.groupby(
        ["model", "stage", "layer"], sort=False
    ):
        group = group.sort_values("seq_len")
        if len(group) < 2:
            continue
        beta, intercept = np.polyfit(
            np.log(group["seq_len"].astype(float)),
            np.log(group["sequence_frobenius_mean"].astype(float)),
            1,
        )
        rows.append(
            {
                "model": model_name,
                "model_label": group.iloc[0]["model_label"],
                "stage": stage,
                "layer": layer,
                "hidden_dim": int(group.iloc[0]["hidden_dim"]),
                "num_layers": int(group.iloc[0]["num_layers"]),
                "num_sequence_lengths": len(group),
                "frobenius_sequence_length_exponent": float(beta),
                "log_intercept": float(intercept),
                "factor_min_across_n": float(group["frobenius_over_sqrt_n_mean"].min()),
                "factor_max_across_n": float(group["frobenius_over_sqrt_n_mean"].max()),
                "factor_max_over_min_across_n": float(
                    group["frobenius_over_sqrt_n_mean"].max()
                    / group["frobenius_over_sqrt_n_mean"].min()
                ),
            }
        )
    return pd.DataFrame(rows)


def depth_verdict(pd, aggregate, scaling):
    rows = []
    for (model_name, seq_len), group in aggregate[
        aggregate["stage"] == "attention_input"
    ].groupby(["model", "seq_len"], sort=False):
        trajectory = group.sort_values("layer")
        first = float(trajectory.iloc[0]["frobenius_over_sqrt_n_mean"])
        final = float(trajectory.iloc[-1]["frobenius_over_sqrt_n_mean"])
        minimum = float(trajectory["frobenius_over_sqrt_n_mean"].min())
        maximum = float(trajectory["frobenius_over_sqrt_n_mean"].max())
        if scaling.empty:
            max_sequence_length_exponent = float("nan")
        else:
            model_scaling = scaling[
                (scaling["model"] == model_name)
                & (scaling["stage"] == "attention_input")
            ]
            max_sequence_length_exponent = float(
                model_scaling["frobenius_sequence_length_exponent"].max()
            )
        rows.append(
            {
                "model": model_name,
                "model_label": trajectory.iloc[0]["model_label"],
                "seq_len": seq_len,
                "hidden_dim": int(trajectory.iloc[0]["hidden_dim"]),
                "num_layers": int(trajectory.iloc[0]["num_layers"]),
                "first_layer_frobenius_over_sqrt_n": first,
                "final_layer_frobenius_over_sqrt_n": final,
                "minimum_frobenius_over_sqrt_n": minimum,
                "maximum_frobenius_over_sqrt_n": maximum,
                "final_over_first": final / first,
                "maximum_over_minimum": maximum / minimum,
                "maximum_over_sqrt_d": maximum
                / math.sqrt(float(trajectory.iloc[0]["hidden_dim"])),
                "max_sequence_length_exponent_over_layers": max_sequence_length_exponent,
            }
        )
    return pd.DataFrame(rows)


def plot_model(ax, aggregate, model_name, show_title=True):
    model_data = aggregate[
        (aggregate["model"] == model_name)
        & (aggregate["stage"] == "attention_input")
    ]
    plotted_changes = []
    for seq_len, group in model_data.groupby("seq_len", sort=True):
        group = group.sort_values("layer")
        baseline = float(group.iloc[0]["frobenius_over_sqrt_n_mean"])
        changes = 100.0 * (
            group["frobenius_over_sqrt_n_mean"].astype(float) / baseline - 1.0
        )
        plotted_changes.extend(changes.tolist())
        ax.plot(
            group["layer"],
            changes,
            marker="o",
            markersize=3.0,
            linewidth=1.5,
            label=f"N={seq_len}",
        )
    if show_title:
        ax.set_title(model_label(model_name))
    ax.set_xlabel(r"Layer $l$")
    ax.set_ylabel(r"$L^2$ norm change (%)")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
    lower = min(plotted_changes)
    upper = max(plotted_changes)
    span = max(upper - lower, 1.0)
    ax.set_ylim(lower - 0.2 * span, upper + 0.3 * span)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def model_slug(model_name):
    return "".join(
        character.lower() if character.isalnum() else "-"
        for character in model_name
    ).strip("-")


def save_figure(plt, aggregate, output_path):
    models = aggregate["model"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(6.4 * len(models), 4.5),
        squeeze=False,
        sharey=False,
    )
    for ax, model_name in zip(axes[0], models):
        plot_model(ax, aggregate, model_name)
    fig.tight_layout()
    output_path = ensure_parent(output_path)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"), dpi=220)
    plt.close(fig)

    for model_name in models:
        model_figure, model_axis = plt.subplots(figsize=(6.4, 4.5))
        plot_model(model_axis, aggregate, model_name, show_title=False)
        model_figure.tight_layout()
        model_output = output_path.with_name(
            f"{output_path.stem}_{model_slug(model_name)}{output_path.suffix}"
        )
        model_figure.savefig(model_output, dpi=220)
        model_figure.savefig(model_output.with_suffix(".pdf"), dpi=220)
        plt.close(model_figure)


def main():
    args = parse_args()
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    if any(seq_len < 1 for seq_len in args.seq_lens):
        raise ValueError("all sequence lengths must be positive")
    seed_everything(args.seed)

    all_raw = []
    all_samples = []
    all_summary = []
    revisions = {}
    for model_name in args.models:
        raw_frames, sample_rows, summary_rows, revision = collect_model(
            torch,
            np,
            load_dataset,
            AutoModel,
            AutoTokenizer,
            args,
            model_name,
        )
        all_raw.extend(raw_frames)
        all_samples.extend(sample_rows)
        all_summary.extend(summary_rows)
        revisions[model_name] = revision

    raw = raw_dataframe(pd, all_raw)
    samples = pd.DataFrame(all_samples)
    token_summary = pd.DataFrame(all_summary)
    aggregate = aggregate_samples(pd, samples)
    summary = aggregate.merge(
        token_summary,
        on=[
            "model",
            "model_label",
            "seq_len",
            "stage",
            "layer",
            "hidden_dim",
            "num_layers",
        ],
        how="left",
        validate="one_to_one",
    )
    scaling = sequence_scaling(pd, np, aggregate)
    verdict = depth_verdict(pd, aggregate, scaling)

    raw_output = ensure_parent(args.raw_output)
    sample_output = ensure_parent(args.sample_output)
    summary_output = ensure_parent(args.summary_output)
    scaling_output = ensure_parent(args.scaling_output)
    verdict_output = ensure_parent(args.verdict_output)
    raw.to_parquet(raw_output, index=False)
    samples.to_csv(sample_output, index=False)
    summary.to_csv(summary_output, index=False)
    scaling.to_csv(scaling_output, index=False)
    verdict.to_csv(verdict_output, index=False)
    save_figure(plt, aggregate, Path(args.figure_output))

    write_json(
        metadata_path(raw_output),
        {
            **now_metadata(),
            "models": args.models,
            "model_revisions": revisions,
            "architecture": "post-residual LayerNorm encoder or decoder",
            "dataset": "wikitext",
            "config": "wikitext-103-raw-v1",
            "split": "test",
            "sequence_lengths": args.seq_lens,
            "num_examples_per_sequence_length": args.max_samples,
            "shared_max_token_budget": args.max_tokens
            or max(args.seq_lens) * args.max_samples,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
            "definition": (
                "The primary factor is the exact per-sample Frobenius norm of "
                "the sequence matrix entering each attention layer, divided by "
                "sqrt(valid tokens) and recorded as attention_input. This includes "
                "the embedding stream entering the first attention layer and the "
                "preceding normalized block output at later layers. "
                "frobenius_over_sqrt_nd is a dimension-normalized comparison "
                "only; it does not rerun the model with rescaled LayerNorm gains. "
                "sequence_spectral_power is an "
                "eight-step power-iteration cross-check, not an exact SVD."
            ),
            "raw_output": str(raw_output),
            "sample_output": str(sample_output),
            "summary_output": str(summary_output),
            "scaling_output": str(scaling_output),
            "verdict_output": str(verdict_output),
            "figure_output": str(args.figure_output),
            "package_versions": package_versions(),
        },
    )
    print(verdict.to_string(index=False), flush=True)
    print(f"wrote {raw_output}", flush=True)
    print(f"wrote {sample_output}", flush=True)
    print(f"wrote {summary_output}", flush=True)
    print(f"wrote {scaling_output}", flush=True)
    print(f"wrote {verdict_output}", flush=True)
    print(f"wrote {args.figure_output}", flush=True)


if __name__ == "__main__":
    main()
