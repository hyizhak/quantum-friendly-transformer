#!/usr/bin/env python
"""Generate companion LayerNorm denominator figures with median and p5 curves."""

from __future__ import annotations

import argparse
from pathlib import Path



import matplotlib.pyplot as plt
import pandas as pd

from quantum_friendly_transformer.experiments.lib.plot_style import (
    ANNOTATION_FONT_SIZE,
    FIGURE_SIZE,
    GRID_ALPHA,
    LEGEND_FONT_SIZE,
    LINE_WIDTH,
    LOG_X_TICK_ROTATION,
    MARKER_SIZE,
    apply_plot_style,
    save_figure,
)


DEFAULT_AGGREGATE_DIR = Path("experiments/outputs/aggregate_dscale_layernorm")
DEFAULT_OUTPUT_DIR = Path("experiments/outputs/figures")
NORM_LABELS = {"norm1": "Norm 1", "norm2": "Norm 2"}
NORM_MARKERS = {"norm1": "o", "norm2": "s"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, default=DEFAULT_AGGREGATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def exact_d_ticks(ax, values) -> None:
    ticks = sorted({int(value) for value in values})
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [str(value) for value in ticks],
        rotation=LOG_X_TICK_ROTATION,
        ha="right",
    )


def plot_median_and_p5(ax, frame, *, label: str, marker: str, color=None):
    (median_line,) = ax.plot(
        frame["d"],
        frame["median"],
        color=color,
        marker=marker,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label=f"{label} median",
    )
    ax.plot(
        frame["d"],
        frame["p5"],
        color=median_line.get_color(),
        marker=marker,
        linewidth=max(LINE_WIDTH - 0.2, 1.0),
        markersize=max(MARKER_SIZE - 0.8, 3.0),
        linestyle="--",
        alpha=0.78,
        label=f"{label} p5",
    )
    return median_line


def annotate_models(ax, frame) -> None:
    for index, (_, row) in enumerate(frame.sort_values("d").iterrows()):
        ax.annotate(
            str(row["model"]).split("/")[-1],
            (row["d"], row["median"]),
            textcoords="offset points",
            xytext=(0, 9 if index % 2 == 0 else -16),
            ha="center",
            fontsize=ANNOTATION_FONT_SIZE,
        )


def style_axes(ax, frame, *, title: str | None = None) -> None:
    ax.set_xscale("log", base=2)
    exact_d_ticks(ax, frame["d"])
    ax.set_xlabel("Embedding dimension d")
    ax.set_ylabel("LayerNorm denominator")
    if title:
        ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", alpha=GRID_ALPHA)
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)


def save_combined(plt, frame, output_dir: Path, stem: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for norm_slot, slot_frame in frame.groupby("norm_slot", sort=True):
        slot_frame = slot_frame.sort_values("d")
        plot_median_and_p5(
            ax,
            slot_frame,
            label=NORM_LABELS[norm_slot],
            marker=NORM_MARKERS[norm_slot],
        )
    annotation_rows = frame.sort_values("median").drop_duplicates("model", keep="last")
    annotate_models(ax, annotation_rows)
    style_axes(ax, frame)
    outputs = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"figure_layernorm_factor_vs_d_{stem}_median_p5.{suffix}"
        save_figure(fig, path)
        outputs.append(path)
    plt.close(fig)
    return outputs


def save_norm_slot(plt, frame, output_dir: Path, stem: str, norm_slot: str) -> list[Path]:
    frame = frame.sort_values("d")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    plot_median_and_p5(
        ax,
        frame,
        label=NORM_LABELS[norm_slot],
        marker=NORM_MARKERS[norm_slot],
    )
    annotate_models(ax, frame)
    style_axes(ax, frame, title=NORM_LABELS[norm_slot])
    outputs = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"figure_layernorm_factor_vs_d_{stem}_{norm_slot}_median_p5.{suffix}"
        save_figure(fig, path)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    apply_plot_style(plt)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.aggregate_dir.glob("layernorm_factor_vs_d_*_by_norm.csv"))
    if not paths:
        raise FileNotFoundError(f"no LayerNorm by-norm aggregates found in {args.aggregate_dir}")

    outputs = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {"model", "d", "norm_slot", "median", "p5"}
        if frame.empty or not required.issubset(frame.columns):
            missing = sorted(required - set(frame.columns))
            raise ValueError(f"{path} is missing required columns: {missing}")
        frame = frame[frame["model"] != "distilgpt2"].copy()
        stem = path.stem.removeprefix("layernorm_factor_vs_d_").removesuffix("_by_norm")
        outputs.extend(save_combined(plt, frame, args.output_dir, stem))
        for norm_slot, slot_frame in frame.groupby("norm_slot", sort=True):
            outputs.extend(save_norm_slot(plt, slot_frame, args.output_dir, stem, norm_slot))

    print(f"wrote {len(outputs)} median+p5 LayerNorm denominator figures to {args.output_dir}")


if __name__ == "__main__":
    main()
