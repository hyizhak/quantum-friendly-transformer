#!/usr/bin/env python
"""Plot median Z_j/N curves for selected stable, well-known model families."""

import argparse
from pathlib import Path



import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import pandas as pd

from quantum_friendly_transformer.experiments.lib.plot_style import (
    FIGURE_SIZE,
    GRID_ALPHA,
    LEGEND_FONT_SIZE,
    LINE_WIDTH,
    LOG_X_TICK_ROTATION,
    MARKER_SIZE,
    apply_plot_style,
    save_figure,
)


AGGREGATE_DIR = Path("experiments/outputs/aggregate_n_scaling_32_8192")
FIGURE_DIR = Path("experiments/outputs/figures_30_8192")

MODEL_LABELS = {
    "google/gemma-2b": "Gemma 2B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek-R1-Distill-Qwen 1.5B",
    "mistralai/Mistral-7B-v0.1": "Mistral 7B",
    "microsoft/Phi-3.5-mini-instruct": "Phi-3.5 Mini",
    "Qwen/Qwen2.5-Math-7B-Instruct": "Qwen2.5-Math 7B Instruct",
}

SEQUENCE_LENGTHS = [512, 1024, 2048, 4096, 8192]
Y_TICKS = [0.1, 0.2, 0.5, 1, 2, 5]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, default=AGGREGATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    apply_plot_style(plt)
    constants = pd.read_csv(args.aggregate_dir / "table_N_scaling_extended_constants.csv")
    selected = constants[
        constants["model"].isin(MODEL_LABELS)
        & constants["seq_len"].isin(SEQUENCE_LENGTHS)
    ].copy()

    counts = selected.groupby("model")["seq_len"].nunique()
    missing = [model for model in MODEL_LABELS if counts.get(model, 0) != len(SEQUENCE_LENGTHS)]
    if missing:
        raise ValueError(f"models missing complete sequence-length curves: {missing}")

    summary_rows = []
    for model in MODEL_LABELS:
        curve = selected[selected["model"] == model].sort_values("seq_len")
        values = curve["r_median"].astype(float)
        summary_rows.append(
            {
                "model": model,
                "smallest_N_median_Z_over_N": values.iloc[0],
                "largest_N_median_Z_over_N": values.iloc[-1],
                "final_over_smallest_N": values.iloc[-1] / values.iloc[0],
                "curve_min": values.min(),
                "curve_max": values.max(),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        args.aggregate_dir / "table_stable_famous_families_within_plot_range.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    markers = ["o", "s", "^", "D", "P"]
    for marker, (model, label) in zip(markers, MODEL_LABELS.items(), strict=True):
        curve = selected[selected["model"] == model].sort_values("seq_len")
        ax.plot(
            curve["seq_len"],
            curve["r_median"],
            marker=marker,
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=label,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.margins(x=0.05)
    ax.set_ylim(Y_TICKS[0], Y_TICKS[-1])
    ax.xaxis.set_major_locator(FixedLocator(SEQUENCE_LENGTHS))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.tick_params(axis="x", labelrotation=LOG_X_TICK_ROTATION)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel(r"Median $Z_j/N$ (log scale)")
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    ax.grid(True, which="both", alpha=GRID_ALPHA)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stems = [
        "figure_Z_over_N_stable_famous_families",
        "figure_30_8192_raw_median_final_gt_25pct_of_model_max_no_bloom_no_gemma3_1b",
    ]
    for stem in stems:
        save_figure(fig, args.output_dir / f"{stem}.png")
        save_figure(fig, args.output_dir / f"{stem}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
