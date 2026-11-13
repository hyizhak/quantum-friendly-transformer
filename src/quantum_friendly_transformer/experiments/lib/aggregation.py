"""Reusable aggregation helpers for forward diagnostic parquet outputs."""

from __future__ import annotations


ATTENTION_METRICS = ["r", "softmax_overhead", "effective_support_ratio", "top1_mass"]
NORM_METRICS = ["ln1_denom", "ln2_denom", "rms_denom", "rms2_denom"]


def quantile_table(df, group_cols, metrics):
    frames = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        subset = df[group_cols + [metric]].dropna(subset=[metric])
        if subset.empty:
            continue
        grouped = subset.groupby(group_cols)[metric]
        out = grouped.quantile([0.01, 0.05, 0.50, 0.95]).unstack()
        out = out.rename(columns={0.01: "p1", 0.05: "p5", 0.50: "median", 0.95: "p95"}).reset_index()
        out["metric"] = metric
        out["sample_count"] = grouped.count().to_numpy()
        frames.append(out)
    if not frames:
        return None
    return __import__("pandas").concat(frames, ignore_index=True)


def hidden_growth(df):
    import pandas as pd

    needed = {"model", "dataset", "task", "seq_len", "layer", "hidden_fro_over_sqrtN", "hidden_p95_row_norm", "hidden_spectral_over_sqrtN"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    layer_stats = (
        df.groupby(["model", "dataset", "task", "seq_len", "layer"])[
            ["hidden_fro_over_sqrtN", "hidden_p95_row_norm", "hidden_spectral_over_sqrtN"]
        ]
        .median()
        .reset_index()
    )
    rows = []
    for keys, group in layer_stats.groupby(["model", "dataset", "task", "seq_len"]):
        group = group.sort_values("layer")
        first_fro = float(group.iloc[0]["hidden_fro_over_sqrtN"])
        first_row = float(group.iloc[0]["hidden_p95_row_norm"])
        rows.append(
            {
                "model": keys[0],
                "dataset": keys[1],
                "task": keys[2],
                "seq_len": keys[3],
                "max_hidden_fro_over_sqrtN": float(group["hidden_fro_over_sqrtN"].max()),
                "median_hidden_fro_over_sqrtN": float(group["hidden_fro_over_sqrtN"].median()),
                "max_hidden_spectral_over_sqrtN": float(group["hidden_spectral_over_sqrtN"].max()),
                "growth_F": float(group["hidden_fro_over_sqrtN"].max() / first_fro) if first_fro else float("nan"),
                "growth_row": float(group["hidden_p95_row_norm"].max() / first_row) if first_row else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def resource_constants(df, attention, norms, growth):
    import numpy as np
    import pandas as pd

    group_cols = ["model", "dataset", "task", "seq_len"]
    config_cols = ["hidden_dim", "num_layers", "attention_type", "norm_type"]
    config = df[group_cols + [c for c in config_cols if c in df.columns]].drop_duplicates(group_cols)

    def metric_slice(table, metric):
        if table is None or table.empty:
            return pd.DataFrame(columns=group_cols)
        out = table[table["metric"] == metric].copy()
        if out.empty:
            return pd.DataFrame(columns=group_cols)
        collapsed = (
            out.groupby(group_cols)
            .agg({"p5": "min", "median": "median", "p95": "max", "sample_count": "sum"})
            .reset_index()
        )
        return collapsed.rename(
            columns={
                "p5": f"{metric}_p5",
                "median": f"{metric}_median",
                "p95": f"{metric}_p95",
                "sample_count": f"{metric}_sample_count",
            }
        )

    out = config.copy()
    for metric in ["r", "softmax_overhead", "effective_support_ratio", *NORM_METRICS]:
        source = attention if metric in ATTENTION_METRICS else norms
        out = out.merge(metric_slice(source, metric), on=group_cols, how="left")
    if growth is not None and not growth.empty:
        out = out.merge(growth, on=group_cols, how="left")
    if "r_p5" in out.columns:
        out["softmax_overhead_from_p5_r"] = 1.0 / np.sqrt(out["r_p5"])
    denom_cols = [c for c in ["ln1_denom_p5", "ln2_denom_p5", "rms_denom_p5", "rms2_denom_p5"] if c in out.columns]
    out["normalisation_denom_p5"] = out[denom_cols].min(axis=1) if denom_cols else np.nan
    return out
