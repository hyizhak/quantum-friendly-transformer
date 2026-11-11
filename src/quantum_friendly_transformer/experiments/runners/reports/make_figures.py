import argparse
from pathlib import Path

from quantum_friendly_transformer.experiments.lib.plot_style import (
    ANNOTATION_FONT_SIZE,
    FIGURE_SIZE,
    GRID_ALPHA,
    LEGEND_FONT_SIZE,
    LINE_WIDTH,
    LOG_X_TICK_ROTATION,
    MARKER_SIZE,
    SAVE_DPI,
    apply_plot_style,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Make figures for NLP diagnostic aggregates.")
    parser.add_argument("--aggregate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv_or_empty(pd, path):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def save_norm_denoms(pd, plt, aggregate_dir, output_dir):
    path = aggregate_dir / "table_A1_norm_quantiles.csv"
    if not path.exists():
        return
    df = read_csv_or_empty(pd, path)
    if df.empty or "metric" not in df.columns:
        return
    for (model, task, seq_len), group in df.groupby(["model", "task", "seq_len"]):
        safe = f"{model}_{task}_{seq_len}".replace("/", "_")
        slot_by_metric = {
            "ln1_denom": "norm1",
            "rms_denom": "norm1",
            "ln2_denom": "norm2",
            "rms2_denom": "norm2",
        }
        group = group.assign(norm_slot=group["metric"].map(slot_by_metric))
        group = group.dropna(subset=["norm_slot"])
        for norm_slot, slot_group in group.groupby("norm_slot", sort=True):
            fig, ax = plt.subplots(figsize=(8, 4))
            for metric, sub in slot_group.groupby("metric"):
                sub = sub.sort_values("layer")
                ax.plot(sub["layer"], sub["p5"], label=metric)
            ax.set_xlabel("Layer")
            metrics = set(slot_group["metric"].dropna())
            if metrics and all(str(metric).startswith("rms") for metric in metrics):
                ylabel = "RMSNorm denominator"
            elif metrics and all(str(metric).startswith("ln") for metric in metrics):
                ylabel = "LayerNorm denominator"
            else:
                ylabel = "LayerNorm/RMSNorm denominator"
            ax.set_ylabel(f"{ylabel} p5")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / f"figure_A2_{norm_slot}_denoms_{safe}.png", dpi=200)
            plt.close(fig)

        # A split regeneration supersedes the legacy figure that overlaid both
        # normalization slots.
        (output_dir / f"figure_A2_norm_denoms_{safe}.png").unlink(missing_ok=True)


def save_hidden_growth(pd, plt, aggregate_dir, output_dir):
    path = aggregate_dir / "table_C1_hidden_growth.csv"
    if not path.exists():
        return
    df = read_csv_or_empty(pd, path)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = (df["model"] + "\n" + df["task"].fillna("") + " N=" + df["seq_len"].astype(str)).tolist()
    ax.bar(range(len(df)), df["growth_F"])
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("max_l Frobenius growth")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_C1_hidden_growth.png", dpi=200)
    plt.close(fig)


def short_model_name(model):
    return str(model).split("/")[-1]


def exact_d_ticks(ax, values, rotation=0):
    ticks = sorted({int(value) for value in values})
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(value) for value in ticks], rotation=rotation, ha="right" if rotation else "center")


def qwen_model_label(models):
    return ", ".join(short_model_name(model) for model in str(models).split(";") if model)


def filter_norm_factor_rows(df):
    if "model" not in df.columns:
        return df
    return df[df["model"] != "distilgpt2"].copy()


def plot_median_and_p5(ax, x, median, p5, *, label=None, color=None, marker="o", linewidth=LINE_WIDTH, markersize=MARKER_SIZE):
    (line,) = ax.plot(
        x,
        median,
        marker=marker,
        linewidth=linewidth,
        markersize=markersize,
        color=color,
        label=f"{label} median" if label else "median",
    )
    ax.plot(
        x,
        p5,
        marker=marker,
        linewidth=max(linewidth - 0.2, 1.0),
        markersize=max(markersize - 0.8, 3.0),
        linestyle="--",
        color=line.get_color(),
        alpha=0.78,
        label=f"{label} p5" if label else "p5",
    )
    return line


def plot_median(ax, x, median, *, label=None, color=None, marker="o", linewidth=LINE_WIDTH, markersize=MARKER_SIZE):
    (line,) = ax.plot(
        x,
        median,
        marker=marker,
        linewidth=linewidth,
        markersize=markersize,
        color=color,
        label=f"{label} median" if label else "median",
    )
    return line


def save_single_norm_factor_figure(
    plt,
    df,
    output_dir,
    filename_stem,
    ylabel,
    figsize=FIGURE_SIZE,
    include_p5=True,
    ylim_bottom=None,
):
    df = df.sort_values("d").reset_index(drop=True)
    norm_slot = df["norm_slot"].iloc[0]
    norm_label = {"norm1": "Norm 1", "norm2": "Norm 2"}[norm_slot]
    fig, ax = plt.subplots(figsize=figsize)
    if include_p5:
        plot_median_and_p5(
            ax,
            df["d"],
            df["median"],
            df["p5"],
            linewidth=1.8,
            markersize=4.8,
        )
        minimum_row = df.loc[df["p5"].idxmin()]
        minimum_p5 = float(minimum_row["p5"])
        minimum_label = f"{minimum_p5:.3g}"
        ax.axhline(
            minimum_p5,
            color="tab:red",
            linestyle=":",
            linewidth=1.4,
            alpha=0.9,
            label=f"minimum p5 = {minimum_label}",
        )
        ax.scatter(
            [minimum_row["d"]],
            [minimum_p5],
            color="tab:red",
            marker="*",
            s=90,
            zorder=5,
        )
        ax.annotate(
            minimum_label,
            (minimum_row["d"], minimum_p5),
            textcoords="offset points",
            xytext=(7, 7),
            color="tab:red",
            fontsize=LEGEND_FONT_SIZE,
            fontweight="bold",
        )
    else:
        plot_median(
            ax,
            df["d"],
            df["median"],
            linewidth=1.8,
            markersize=4.8,
        )
    for idx, row in df.iterrows():
        model = row["models"] if "models" in df.columns else row["model"]
        ax.annotate(
            qwen_model_label(model) if "models" in df.columns else short_model_name(model),
            (row["d"], row["median"]),
            textcoords="offset points",
            xytext=(0, 10 if idx % 2 == 0 else -18),
            ha="center",
            fontsize=ANNOTATION_FONT_SIZE,
        )
    ax.set_xscale("log", base=2)
    exact_d_ticks(ax, df["d"], rotation=LOG_X_TICK_ROTATION)
    ax.set_xlabel("Embedding dimension d")
    ax.set_ylabel(ylabel)
    ax.set_title(norm_label)
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    ax.grid(True, which="both", alpha=GRID_ALPHA)
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{filename_stem}_{norm_slot}.{suffix}", dpi=SAVE_DPI)
    plt.close(fig)


def save_layernorm_factor_vs_d(pd, plt, aggregate_dir, output_dir, stems=None):
    paths = sorted(aggregate_dir.glob("layernorm_factor_vs_d_*.csv"))
    if stems is not None:
        paths = [path for path in paths if path.stem.replace("layernorm_factor_vs_d_", "") in stems]
    for path in paths:
        stem = path.stem.replace("layernorm_factor_vs_d_", "")
        by_layer_path = aggregate_dir / f"layernorm_factor_by_layer_{stem}.csv"
        if not by_layer_path.exists():
            continue
        by_layer = read_csv_or_empty(pd, by_layer_path)
        required = {"model", "d", "slot", "p5", "median", "p95", "sample_count"}
        if by_layer.empty or not required.issubset(by_layer.columns):
            continue
        by_layer = filter_norm_factor_rows(by_layer)
        by_layer["norm_slot"] = by_layer["slot"].map({"ln1": "norm1", "ln2": "norm2"})
        by_layer = by_layer.dropna(subset=["norm_slot"])
        metadata = [
            column
            for column in ["model", "family", "dataset", "task", "seq_len", "d", "num_layers", "norm_slot"]
            if column in by_layer.columns
        ]
        df = (
            by_layer.groupby(metadata, dropna=False)
            .agg(
                p5=("p5", "min"),
                median=("median", "median"),
                p95=("p95", "max"),
                sample_count=("sample_count", "sum"),
            )
            .reset_index()
            .sort_values(["norm_slot", "d", "model"])
        )
        global_path = aggregate_dir / f"layernorm_factor_vs_d_{stem}_by_norm_global.csv"
        if global_path.exists():
            global_df = read_csv_or_empty(pd, global_path)
            global_required = {"model", "d", "norm_slot", "p5", "median", "p95", "sample_count"}
            if not global_df.empty and global_required.issubset(global_df.columns):
                df = filter_norm_factor_rows(global_df).sort_values(["norm_slot", "d", "model"])
        if df.empty:
            continue
        df.to_csv(aggregate_dir / f"layernorm_factor_vs_d_{stem}_by_norm.csv", index=False)

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        for norm_slot, slot_group in df.groupby("norm_slot", sort=True):
            slot_group = slot_group.sort_values("d")
            plot_median(
                ax,
                slot_group["d"],
                slot_group["median"],
                label={"norm1": "Norm 1", "norm2": "Norm 2"}[norm_slot],
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
            )
        annotation_rows = df.sort_values("median").drop_duplicates("model", keep="last")
        for _, row in annotation_rows.iterrows():
            ax.annotate(
                short_model_name(row["model"]),
                (row["d"], row["median"]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=ANNOTATION_FONT_SIZE,
            )
        ax.set_xscale("log", base=2)
        exact_d_ticks(ax, df["d"], rotation=LOG_X_TICK_ROTATION)
        ax.set_xlabel("Embedding dimension d")
        ax.set_ylabel("LayerNorm denominator")
        ax.set_ylim(bottom=0)
        ax.grid(True, which="both", alpha=GRID_ALPHA)
        ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"figure_layernorm_factor_vs_d_{stem}.{suffix}", dpi=SAVE_DPI)
        plt.close(fig)
        for _, slot_group in df.groupby("norm_slot", sort=True):
            save_single_norm_factor_figure(
                plt,
                slot_group,
                output_dir,
                f"figure_layernorm_factor_vs_d_{stem}",
                "LayerNorm denominator",
                include_p5=False,
                ylim_bottom=0,
            )


def load_layernorm_family_series(pd, aggregate_dir, filename, label):
    stem = Path(filename).stem.replace("layernorm_factor_vs_d_", "")
    candidates = [
        aggregate_dir / f"layernorm_factor_vs_d_{stem}_by_norm_global.csv",
        aggregate_dir / f"layernorm_factor_vs_d_{stem}_by_norm.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return pd.DataFrame()
    df = read_csv_or_empty(pd, path)
    required = {"model", "d", "norm_slot", "median", "p5", "p95"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "series": label,
            "norm_type": "LayerNorm",
            "norm_slot": df["norm_slot"],
            "d": df["d"].astype(int),
            "median": df["median"].astype(float),
            "p5": df["p5"].astype(float),
            "p95": df["p95"].astype(float),
            "models": df["model"],
        }
    )


def load_supplemental_qwen_rmsnorm_rows(pd, aggregate_dir, seq_len=512):
    paths = [
        aggregate_dir.parent / "raw" / "diag_qwen25_0p5b_instruct_wikitext_N512_1024_2048_4096_8192.parquet",
        aggregate_dir.parent / "raw" / "diag_qwen3_0p6b_wikitext_N512_1024_2048_4096_8192.parquet",
        aggregate_dir.parent / "raw" / "diag_qwen25_1p5b_wikitext_N512_1024_2048_4096_8192.parquet",
        aggregate_dir.parent / "raw" / "diag_qwen3_1p7b_wikitext_N512_1024_2048_4096_8192.parquet",
        aggregate_dir.parent / "raw" / "diag_qwen3_4b_wikitext_N512_1024_2048_4096_8192_zonly.parquet",
        aggregate_dir.parent / "raw" / "diag_qwen25_7b_wikitext_N512_1024_2048_4096_8192_zonly.parquet",
        aggregate_dir.parent / "raw" / "diag_qwen3_8b_wikitext_N512_zonly.parquet",
        aggregate_dir.parent / "raw" / "diag_qwen3_32b_wikitext_N512_zonly.parquet",
    ]
    rows = []
    columns = ["model", "task", "seq_len", "hidden_dim", "norm_type", "layer", "rms_denom", "rms2_denom"]
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=columns)
        df = df[(df["seq_len"] == seq_len) & (df["norm_type"] == "RMSNorm")]
        if df.empty:
            continue
        for metric, norm_slot in [("rms_denom", "norm1"), ("rms2_denom", "norm2")]:
            metric_df = df.dropna(subset=[metric])
            if metric_df.empty:
                continue
            layer_quantiles = metric_df.groupby("layer")[metric].quantile([0.50, 0.95]).unstack()
            rows.append(
                {
                    "model": metric_df["model"].iloc[0],
                    "task": metric_df["task"].iloc[0],
                    "seq_len": int(seq_len),
                    "hidden_dim": int(metric_df["hidden_dim"].iloc[0]),
                    "norm_type": "RMSNorm",
                    "norm_slot": norm_slot,
                    "p5": float(metric_df[metric].quantile(0.05)),
                    "median": float(layer_quantiles[0.50].median()),
                    "p95": float(layer_quantiles[0.95].max()),
                }
            )
    return pd.DataFrame(rows)


def load_qwen_rmsnorm_series(pd, aggregate_dir, seq_len=512):
    candidates = [
        aggregate_dir.parent / "aggregate_n_scaling_30_8192" / "resource_constants.csv",
        aggregate_dir.parent / "aggregate_n_scaling_extended" / "resource_constants.csv",
    ]
    resource_path = next((path for path in candidates if path.exists()), None)
    if resource_path is None:
        return pd.DataFrame()
    resources = read_csv_or_empty(pd, resource_path)
    norm_path = resource_path.parent / "table_A1_norm_quantiles.csv"
    norms = read_csv_or_empty(pd, norm_path) if norm_path.exists() else pd.DataFrame()
    resource_required = {"model", "seq_len", "hidden_dim", "norm_type"}
    norm_required = {"model", "seq_len", "metric", "p5", "median", "p95"}
    if (
        resources.empty
        or norms.empty
        or not resource_required.issubset(resources.columns)
        or not norm_required.issubset(norms.columns)
    ):
        return pd.DataFrame()

    configs = resources[
        (resources["seq_len"] == seq_len)
        & (resources["norm_type"] == "RMSNorm")
        & resources["model"].astype(str).str.startswith("Qwen/")
    ][["model", "seq_len", "hidden_dim", "norm_type"]].drop_duplicates("model")
    qwen_norms = norms[
        (norms["seq_len"] == seq_len)
        & norms["model"].astype(str).str.startswith("Qwen/")
        & norms["metric"].isin(["rms_denom", "rms2_denom"])
    ].copy()
    qwen_norms["norm_slot"] = qwen_norms["metric"].map({"rms_denom": "norm1", "rms2_denom": "norm2"})
    qwen = (
        qwen_norms.groupby(["model", "norm_slot"])
        .agg(p5=("p5", "min"), median=("median", "median"), p95=("p95", "max"))
        .reset_index()
        .merge(configs, on="model", how="inner")
    )

    supplemental = load_supplemental_qwen_rmsnorm_rows(pd, aggregate_dir, seq_len=seq_len)
    if not supplemental.empty:
        supplemental_keys = supplemental[["model", "norm_slot"]].drop_duplicates()
        qwen = qwen.merge(supplemental_keys.assign(from_raw=True), on=["model", "norm_slot"], how="left")
        qwen = qwen[qwen["from_raw"].isna()].drop(columns="from_raw")
        qwen = pd.concat([qwen, supplemental], ignore_index=True, sort=False)
    global_path = aggregate_dir / "qwen_rmsnorm_factor_vs_d_wikitext_N512_by_norm_global.csv"
    if global_path.exists():
        global_qwen = read_csv_or_empty(pd, global_path)
        global_required = {"model", "d", "norm_slot", "p5", "median", "p95"}
        if not global_qwen.empty and global_required.issubset(global_qwen.columns):
            global_qwen["hidden_dim"] = global_qwen["d"]
            global_qwen["norm_type"] = "RMSNorm"
            qwen = global_qwen
    if qwen.empty:
        return pd.DataFrame()
    preferred_models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-32B",
    ]
    preference = {model: idx for idx, model in enumerate(preferred_models)}
    qwen["preference"] = qwen["model"].map(preference).fillna(len(preference)).astype(int)
    qwen = qwen.sort_values(["norm_slot", "hidden_dim", "preference", "model"]).drop_duplicates(
        ["norm_slot", "hidden_dim"], keep="first"
    )

    out = pd.DataFrame(
        {
            "series": "Qwen (RMSNorm)",
            "norm_type": "RMSNorm",
            "norm_slot": qwen["norm_slot"],
            "d": qwen["hidden_dim"].astype(int),
            "median": qwen["median"].astype(float),
            "p5": qwen["p5"].astype(float),
            "p95": qwen["p95"].astype(float),
            "models": qwen["model"].astype(str),
        }
    )
    return out[["series", "norm_type", "norm_slot", "d", "median", "p5", "p95", "models"]]


def save_qwen_rmsnorm_factor_vs_d(pd, plt, aggregate_dir, output_dir):
    qwen = load_qwen_rmsnorm_series(pd, aggregate_dir, seq_len=512)
    if qwen.empty:
        return
    qwen = qwen.sort_values(["norm_slot", "d"])
    qwen.to_csv(aggregate_dir / "qwen_rmsnorm_factor_vs_d_wikitext_N512.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for norm_slot, slot_group in qwen.groupby("norm_slot", sort=True):
        slot_group = slot_group.sort_values("d")
        plot_median_and_p5(
            ax,
            slot_group["d"],
            slot_group["median"],
            slot_group["p5"],
            label={"norm1": "Norm 1", "norm2": "Norm 2"}[norm_slot],
            linewidth=1.8,
            markersize=4.8,
        )
    annotation_rows = qwen.sort_values("median").drop_duplicates("d", keep="last").reset_index(drop=True)
    for idx, row in annotation_rows.iterrows():
        ax.annotate(
            qwen_model_label(row["models"]),
            (row["d"], row["median"]),
            textcoords="offset points",
            xytext=(0, 10 if idx % 2 == 0 else -18),
            ha="center",
            fontsize=ANNOTATION_FONT_SIZE,
        )
    ax.set_xscale("log", base=2)
    exact_d_ticks(ax, qwen["d"], rotation=LOG_X_TICK_ROTATION)
    ax.set_xlabel("Embedding dimension d")
    ax.set_ylabel("RMSNorm denominator")
    ax.grid(True, which="both", alpha=GRID_ALPHA)
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"figure_qwen_rmsnorm_factor_vs_d_wikitext_N512.{suffix}", dpi=SAVE_DPI)
    plt.close(fig)
    for _, slot_group in qwen.groupby("norm_slot", sort=True):
        save_single_norm_factor_figure(
            plt,
            slot_group,
            output_dir,
            "figure_qwen_rmsnorm_factor_vs_d_wikitext_N512",
            "RMSNorm denominator",
            figsize=(8.2, 4.8),
        )


def save_norm_factor_vs_d_comparison(pd, plt, aggregate_dir, output_dir):
    layernorm_series = [
        load_layernorm_family_series(pd, aggregate_dir, "layernorm_factor_vs_d_gpt2_wikitext_N512.csv", "GPT-2 (LayerNorm)"),
        load_layernorm_family_series(pd, aggregate_dir, "layernorm_factor_vs_d_opt_wikitext_N512.csv", "OPT (LayerNorm)"),
    ]
    rmsnorm = load_qwen_rmsnorm_series(pd, aggregate_dir, seq_len=512)
    frames = [frame for frame in layernorm_series + [rmsnorm] if not frame.empty]
    if len(frames) < 2:
        return
    combined = pd.concat(frames, ignore_index=True).sort_values(["series", "norm_slot", "d"])
    combined.to_csv(
        aggregate_dir / "norm_factor_vs_d_wikitext_N512_layernorm_rmsnorm.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for (label, norm_slot), group in combined.groupby(["series", "norm_slot"], sort=False):
        group = group.sort_values("d")
        plot_median_and_p5(
            ax,
            group["d"],
            group["median"],
            group["p5"],
            label=f"{label} / {norm_slot.replace('norm', 'Norm ')}",
            linewidth=1.8,
            markersize=MARKER_SIZE,
        )
    ax.set_xscale("log", base=2)
    exact_d_ticks(ax, combined["d"], rotation=35)
    ax.set_xlabel("Embedding dimension d")
    ax.set_ylabel("Normalization denominator")
    ax.grid(True, which="both", alpha=GRID_ALPHA)
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"figure_norm_factor_vs_d_wikitext_N512_layernorm_rmsnorm.{suffix}", dpi=SAVE_DPI)
    plt.close(fig)


def model_order(models):
    preferred = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen1.5-1.8B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "HuggingFaceTB/SmolLM3-3B",
        "unsloth/Llama-3.2-1B-Instruct",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "microsoft/Phi-3-mini-128k-instruct",
        "microsoft/phi-4",
        "google/gemma-3-1b-pt",
        "google/gemma-3-4b-pt",
    ]
    seen = set(models)
    return [model for model in preferred if model in seen] + [model for model in sorted(seen) if model not in preferred]


def save_series_plot(
    plt,
    df,
    value_col,
    output_path,
    ylabel,
    relative=False,
    model_filter=None,
    save_pdf=False,
    order_legend_by_final_value=False,
    secondary_value_col=None,
):
    if df.empty or value_col not in df.columns:
        return
    seqs = sorted(int(value) for value in df["seq_len"].dropna().unique())
    models = model_order(df["model"].dropna().unique())
    if model_filter is not None:
        models = [model for model in models if model_filter(model, df[df["model"] == model])]
    if not models:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.cm.tab20(range(20))
    legend_items = []
    for idx, model in enumerate(models):
        sub = df[df["model"] == model].sort_values("seq_len")
        values = sub[value_col].astype(float)
        if relative:
            base = values.iloc[0]
            if base == 0:
                continue
            values = values / base
        (line,) = ax.plot(
            sub["seq_len"],
            values,
            marker="o",
            linewidth=LINE_WIDTH,
            markersize=4,
            label=model,
            color=colors[idx % len(colors)],
        )
        if secondary_value_col is not None and secondary_value_col in sub.columns:
            secondary_values = sub[secondary_value_col].astype(float)
            if relative:
                secondary_base = secondary_values.iloc[0]
                if secondary_base == 0:
                    secondary_values = secondary_values * float("nan")
                else:
                    secondary_values = secondary_values / secondary_base
            ax.plot(
                sub["seq_len"],
                secondary_values,
                marker="o",
                linewidth=1.3,
                markersize=3.5,
                linestyle="--",
                color=line.get_color(),
                alpha=0.78,
                label="_nolegend_",
            )
        valid_values = values.dropna()
        if not valid_values.empty:
            legend_items.append((float(valid_values.iloc[-1]), model, line))
    p5_handle = None
    if secondary_value_col is not None:
        (p5_handle,) = ax.plot([], [], color="black", linestyle="--", linewidth=1.3, label="p5")
    ax.set_xscale("log", base=2)
    scale_columns = [value_col]
    if secondary_value_col is not None and secondary_value_col in df.columns:
        scale_columns.append(secondary_value_col)
    if all((df[column].dropna() > 0).all() for column in scale_columns):
        ax.set_yscale("log")
    elif secondary_value_col is not None and all((df[column].dropna() >= 0).all() for column in scale_columns):
        ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_xticks(seqs)
    ax.set_xticklabels([str(seq) for seq in seqs])
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=GRID_ALPHA)
    if order_legend_by_final_value:
        legend_items.sort(key=lambda item: (-item[0], item[1]))
        ax.legend(
            [item[2] for item in legend_items] + ([p5_handle] if p5_handle is not None else []),
            [item[1] for item in legend_items] + (["p5"] if p5_handle is not None else []),
            fontsize=6,
            loc="best",
            frameon=False,
        )
    else:
        ax.legend(fontsize=6, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI)
    if save_pdf:
        fig.savefig(output_path.with_suffix(".pdf"), dpi=SAVE_DPI)
    plt.close(fig)


def selected_model_filter(models, additional_filter=None):
    selected = set(models)
    return lambda model, sub: model in selected and (additional_filter is None or additional_filter(model, sub))


def exclude_qwen3_and_gemma3(model, _sub):
    normalized = str(model).lower().replace("-", "").replace("_", "")
    return "qwen3" not in normalized and "gemma3" not in normalized


def save_filtered_30_8192_plots(pd, plt, aggregate_dir, output_dir, constants, centered):
    median_selector = aggregate_dir / "table_30_model_raw_median_final_gt_25pct_of_model_max_no_bloom_no_gemma3_1b.csv"
    if median_selector.exists() and not constants.empty:
        selected = read_csv_or_empty(pd, median_selector)["model"].dropna().tolist()
        save_series_plot(
            plt,
            constants,
            "r_median",
            output_dir / "figure_30_8192_raw_median_final_gt_25pct_of_model_max_no_bloom_no_gemma3_1b.png",
            "Median and p5 Z / N",
            secondary_value_col="r_p5",
            model_filter=selected_model_filter(selected, exclude_qwen3_and_gemma3),
            save_pdf=True,
            order_legend_by_final_value=True,
        )

    centered_selector = aggregate_dir / "table_30_model_max_shifted_mean_final_gt_25pct_of_model_max_no_bloom_no_gemma3_1b.csv"
    if centered_selector.exists() and not centered.empty:
        selected = read_csv_or_empty(pd, centered_selector)["model"].dropna().tolist()
        save_series_plot(
            plt,
            centered,
            "r_max_centered_mean",
            output_dir / "figure_30_8192_max_shifted_mean_final_gt_25pct_of_model_max_no_bloom_no_gemma3_1b.png",
            "Mean max-centered Z / valid context",
            model_filter=selected_model_filter(selected),
            save_pdf=True,
        )


def save_decoder_summaries(pd, plt, aggregate_dir, output_dir):
    constants_path = aggregate_dir / "table_N_scaling_extended_constants.csv"
    constants = pd.DataFrame()
    if constants_path.exists():
        constants = read_csv_or_empty(pd, constants_path)
        save_series_plot(
            plt,
            constants,
            "r_median",
            output_dir / "figure_N_scaling_extended_decoder_Z_over_valid_context_absolute.png",
            "Median and p5 Z / N",
            model_filter=exclude_qwen3_and_gemma3,
            secondary_value_col="r_p5",
            order_legend_by_final_value=True,
        )
        save_series_plot(
            plt,
            constants,
            "r_median",
            output_dir / "figure_N_scaling_extended_decoder_Z_over_valid_context_relative.png",
            "Median and p5 Z / valid context, normalized to first N",
            model_filter=exclude_qwen3_and_gemma3,
            secondary_value_col="r_p5",
            relative=True,
            order_legend_by_final_value=True,
        )

    mean_path = aggregate_dir / "table_N_scaling_extended_mean_constants.csv"
    if mean_path.exists():
        mean_df = read_csv_or_empty(pd, mean_path)
        save_series_plot(
            plt,
            mean_df,
            "r_mean",
            output_dir / "figure_N_scaling_extended_decoder_mean_Z_over_valid_context_all_models.png",
            "Mean Z / valid context",
        )
        save_series_plot(
            plt,
            mean_df,
            "r_mean",
            output_dir / "figure_N_scaling_extended_decoder_mean_Z_over_valid_context_moderate_models.png",
            "Mean Z / valid context",
            model_filter=lambda _model, sub: float(sub["r_mean"].max()) <= 1000.0,
        )

    trimmed_path = aggregate_dir / "table_N_scaling_extended_upper5pct_trimmed_mean_constants.csv"
    if trimmed_path.exists():
        trimmed = read_csv_or_empty(pd, trimmed_path)
        save_series_plot(
            plt,
            trimmed,
            "r_trimmed_mean",
            output_dir / "figure_N_scaling_extended_decoder_trimmed_mean_Z_over_valid_context_all_models.png",
            "Mean Z / valid context after dropping top 5% log_r",
        )

    centered = pd.DataFrame()
    centered_path = aggregate_dir / "table_N_scaling_extended_max_centered_constants.csv"
    if centered_path.exists():
        centered = read_csv_or_empty(pd, centered_path)
        save_series_plot(
            plt,
            centered,
            "r_max_centered_mean",
            output_dir / "figure_N_scaling_extended_decoder_max_centered_mean_Z_over_valid_context_all_models.png",
            "Mean max-centered Z / valid context",
        )

    save_filtered_30_8192_plots(pd, plt, aggregate_dir, output_dir, constants, centered)


def main():
    args = parse_args()
    import pandas as pd
    import matplotlib.pyplot as plt

    apply_plot_style(plt)

    aggregate_dir = Path(args.aggregate_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_norm_denoms(pd, plt, aggregate_dir, output_dir)
    save_hidden_growth(pd, plt, aggregate_dir, output_dir)
    save_layernorm_factor_vs_d(pd, plt, aggregate_dir, output_dir)
    save_qwen_rmsnorm_factor_vs_d(pd, plt, aggregate_dir, output_dir)
    save_norm_factor_vs_d_comparison(pd, plt, aggregate_dir, output_dir)
    save_decoder_summaries(pd, plt, aggregate_dir, output_dir)
    print(f"wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
