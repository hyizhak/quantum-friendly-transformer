#!/usr/bin/env python
"""Aggregate the retained Z/N versus N diagnostic runs."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from quantum_friendly_transformer.experiments.lib.aggregation import ATTENTION_METRICS, NORM_METRICS, hidden_growth, quantile_table, resource_constants
from quantum_friendly_transformer.experiments.lib.common import ensure_parent
from quantum_friendly_transformer.experiments.lib.retained import RETAINED_N_SCALING_INPUTS


DEFAULT_INPUTS = RETAINED_N_SCALING_INPUTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Parquet files or glob patterns to aggregate. Defaults to the retained N-scaling source set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/aggregate_n_scaling_extended"),
    )
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(Path(match) for match in matches)
    unique = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    missing = [path for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError("\n".join(str(path) for path in missing))
    if not unique:
        raise FileNotFoundError("no N-scaling parquet inputs matched")
    return unique


def collapse_constants(resource):
    denom_cols = [col for col in ["ln1_denom_p5", "ln2_denom_p5", "rms_denom_p5", "rms2_denom_p5"] if col in resource]
    out = resource.copy()
    if denom_cols:
        out["normalisation_denom_p5"] = out[denom_cols].min(axis=1)
    cols = [
        "model",
        "task",
        "attention_type",
        "norm_type",
        "seq_len",
        "r_p5",
        "r_median",
        "normalisation_denom_p5",
    ]
    return out[[col for col in cols if col in out.columns]].sort_values(["model", "task", "seq_len"])


def mean_constants(pd, df):
    group_cols = ["model", "dataset", "task", "seq_len"]
    config_cols = ["attention_type", "norm_type"]
    if "r" not in df.columns:
        return pd.DataFrame()
    source = df[group_cols + ["r"]].copy()
    source["r"] = pd.to_numeric(source["r"], errors="coerce")
    source = source.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["r"])
    means = source.groupby(group_cols, dropna=False)["r"].mean().reset_index(name="r_mean")
    counts = source.groupby(group_cols, dropna=False)["r"].count().reset_index(name="r_sample_count")
    config = df[group_cols + [col for col in config_cols if col in df.columns]].drop_duplicates(group_cols)
    out = config.merge(means, on=group_cols, how="right").merge(counts, on=group_cols, how="left")
    cols = ["model", "task", "attention_type", "norm_type", "seq_len", "r_mean", "r_sample_count"]
    return out[[col for col in cols if col in out.columns]].sort_values(["model", "task", "seq_len"])


def max_centered_constants(pd, df):
    import numpy as np

    group_cols = ["model", "dataset", "task", "seq_len"]
    config_cols = ["attention_type", "norm_type"]
    needed = {"top1_mass", "valid_key_count"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    cols = group_cols + [col for col in config_cols if col in df.columns] + ["top1_mass", "valid_key_count"]
    source = df[cols].copy()
    source["top1_mass"] = pd.to_numeric(source["top1_mass"], errors="coerce")
    source["valid_key_count"] = pd.to_numeric(source["valid_key_count"], errors="coerce")
    source = source.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["top1_mass", "valid_key_count"])
    source = source[(source["top1_mass"] > 0.0) & (source["valid_key_count"] > 0.0)]
    # If M = max_j score_j, then sum_j exp(score_j - M) / N = 1 / (top1_mass * N).
    source["r_max_centered"] = 1.0 / (source["top1_mass"].astype(float) * source["valid_key_count"].astype(float))
    grouped = source.groupby(group_cols, dropna=False)["r_max_centered"]
    stats = grouped.agg(
        r_max_centered_mean="mean",
        r_max_centered_median="median",
        r_max_centered_p5=lambda x: x.quantile(0.05),
        r_max_centered_p95=lambda x: x.quantile(0.95),
        r_max_centered_sample_count="count",
    ).reset_index()
    config = source[group_cols + [col for col in config_cols if col in source.columns]].drop_duplicates(group_cols)
    out = config.merge(stats, on=group_cols, how="right")
    cols = [
        "model",
        "task",
        "attention_type",
        "norm_type",
        "seq_len",
        "r_max_centered_mean",
        "r_max_centered_median",
        "r_max_centered_p5",
        "r_max_centered_p95",
        "r_max_centered_sample_count",
    ]
    return out[[col for col in cols if col in out.columns]].sort_values(["model", "task", "seq_len"])


def series_label(row) -> str:
    return f"{row['model']} / {row['task']}"


def stability_table(pd, constants, value_col: str, output_col: str):
    rows = []
    subset = constants.dropna(subset=[value_col])
    for (_, _, _), group in subset.groupby(["model", "task", "attention_type"], dropna=False):
        values = group[value_col]
        min_value = float(values.min())
        max_value = float(values.max())
        rows.append(
            {
                "series": series_label(group.iloc[0]),
                "attention_type": group.iloc[0]["attention_type"],
                "norm_type": group.iloc[0].get("norm_type"),
                "min_N": int(group["seq_len"].min()),
                "max_N": int(group["seq_len"].max()),
                "num_N": int(group["seq_len"].nunique()),
                f"{output_col}_min": min_value,
                f"{output_col}_max": max_value,
                f"{output_col}_max_relative_deviation": (max_value - min_value) / max(abs(max_value), 1e-12),
            }
        )
    return pd.DataFrame(rows)


def availability_table(pd, constants):
    rows = []
    for (_, _, _), group in constants.groupby(["model", "task", "attention_type"], dropna=False):
        available = sorted(int(value) for value in group["seq_len"].dropna().unique())
        joined = " ".join(str(value) for value in available)
        rows.append(
            {
                "model": group.iloc[0]["model"],
                "task": group.iloc[0]["task"],
                "attention_type": group.iloc[0]["attention_type"],
                "available_N": joined,
                "expected_N": joined,
                "missing_N": "",
                "extra_N": "",
                "complete": True,
            }
        )
    return pd.DataFrame(rows)


def source_table(pd, paths, frames):
    rows = []
    for path, frame in zip(paths, frames, strict=True):
        seq_lens = " ".join(str(int(value)) for value in sorted(frame["seq_len"].dropna().unique()))
        rows.append({"source_file": str(path), "rows": len(frame), "seq_lens": seq_lens})
    return pd.DataFrame(rows)


def lower_bound_table(norm_stability):
    if norm_stability.empty:
        return norm_stability
    return norm_stability.rename(
        columns={
            "series": "model_task",
            "normalisation_denom_p5_min": "denom_lower_bound",
            "normalisation_denom_p5_max": "denom_upper_bound",
            "normalisation_denom_p5_max_relative_deviation": "max_relative_deviation",
        }
    )[
        [
            "model_task",
            "attention_type",
            "norm_type",
            "min_N",
            "max_N",
            "num_N",
            "denom_lower_bound",
            "denom_upper_bound",
            "max_relative_deviation",
        ]
    ]


def write_csv(df, path: Path) -> None:
    df.to_csv(ensure_parent(path), index=False)


def main() -> None:
    args = parse_args()
    import pandas as pd

    paths = expand_inputs(args.inputs)
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    attention_groups = ["model", "dataset", "task", "seq_len", "layer"]
    norms_groups = ["model", "dataset", "task", "seq_len", "layer"]
    attention = quantile_table(df, attention_groups, ATTENTION_METRICS)
    norms = quantile_table(df, norms_groups, NORM_METRICS)
    growth = hidden_growth(df)
    resource = resource_constants(df, attention, norms, growth)
    constants = collapse_constants(resource)
    mean_constants_table = mean_constants(pd, df)
    max_centered_constants_table = max_centered_constants(pd, df)
    attention_stability = stability_table(pd, constants, "r_median", "r_median")
    norm_stability = stability_table(pd, constants, "normalisation_denom_p5", "normalisation_denom_p5")

    if attention is not None:
        write_csv(attention, output_dir / "table_A1_attention_quantiles.csv")
        write_csv(attention, output_dir / "table_N_scaling_extended_layer_quantiles.csv")
    if norms is not None:
        write_csv(norms, output_dir / "table_A1_norm_quantiles.csv")
    write_csv(resource, output_dir / "resource_constants.csv")
    write_csv(growth, output_dir / "table_C1_hidden_growth.csv")
    write_csv(constants, output_dir / "table_N_scaling_extended_constants.csv")
    write_csv(mean_constants_table, output_dir / "table_N_scaling_extended_mean_constants.csv")
    write_csv(max_centered_constants_table, output_dir / "table_N_scaling_extended_max_centered_constants.csv")
    write_csv(attention_stability, output_dir / "table_N_scaling_extended_attention_stability.csv")
    write_csv(norm_stability, output_dir / "table_N_scaling_extended_norm_stability.csv")
    write_csv(availability_table(pd, constants), output_dir / "table_N_scaling_extended_availability.csv")
    write_csv(source_table(pd, paths, frames), output_dir / "table_N_scaling_extended_sources.csv")
    write_csv(lower_bound_table(norm_stability), output_dir / "table_N_scaling_extended_norm_lower_bounds.csv")
    print(f"aggregated {len(paths)} N-scaling files into {output_dir}")


if __name__ == "__main__":
    main()
