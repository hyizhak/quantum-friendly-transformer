"""Summarize all completed bounded recovery runs, including unsuccessful runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.lib.experiment import save_csv, save_json
from quantum_friendly_transformer.evidence import EvidenceStore


def summarize(root, config, allow_partial):
    names = list(config["models"])
    expected = [(name, mode, seed) for name in names for mode in config["modes"] for seed in config["seeds"]]
    results, missing = [], []
    for name, mode, seed in expected:
        path = root / f"{name}_{mode}_seed{seed}"
        if not (path / "manifest.json").exists():
            missing.append(path.name)
            continue
        r = json.loads((path / "result.json").read_text())
        if r["smoke"]:
            raise ValueError(f"Smoke output is not a research run: {path}")
        accuracy = r["evaluation"]["correct"] / r["evaluation"]["examples"]
        rows = list(csv.DictReader((path / "attention_inputs.csv").open()))
        results.append({"name": name, "model": r["model"], "mode": mode, "seed": seed,
                        "accuracy": accuracy, "correct": r["evaluation"]["correct"],
                        "examples": r["evaluation"]["examples"], "mean_nll": r["evaluation"]["mean_nll"],
                        "selected_step": r["selected_step"], "selection_accuracy": r["selection_accuracy"],
                        "historical_recovery_threshold": r["historical_recovery_threshold"],
                        "meets_historical_threshold": r["meets_historical_threshold"],
                        "alpha_s": r["alpha_s"], "configured_denominator_floor": r["configured_denominator_floor"],
                        "maximum_matrix_to_alpha_ratio": r["maximum_observed_matrix_to_alpha_ratio"],
                        "matrix_norm_observations": len(rows), "seconds": r["seconds"], "directory": str(path)})
    if missing and not allow_partial:
        raise ValueError("Incomplete sweep: " + ", ".join(missing))
    if not results:
        raise ValueError("No completed runs")
    summary = []
    for name in names:
        for mode in config["modes"]:
            rows = [r for r in results if r["name"] == name and r["mode"] == mode]
            if not rows:
                continue
            acc = np.array([r["accuracy"] for r in rows])
            summary.append({"name": name, "mode": mode, "training_seeds": len(rows),
                            "mean_accuracy": float(acc.mean()),
                            "std_accuracy": float(acc.std(ddof=1)) if len(rows) > 1 else None,
                            "minimum_accuracy": float(acc.min()), "maximum_accuracy": float(acc.max()),
                            "seeds_meeting_historical_threshold": sum(r["meets_historical_threshold"] for r in rows),
                            "historical_recovery_threshold": rows[0]["historical_recovery_threshold"],
                            "maximum_matrix_to_alpha_ratio": max(r["maximum_matrix_to_alpha_ratio"] for r in rows)})
    lookup = {(r["name"], r["mode"], r["seed"]): r for r in results}
    paired = []
    for name in names:
        for seed in config["seeds"]:
            a, b = lookup.get((name, "bounded", seed)), lookup.get((name, "uniform", seed))
            if a is not None and b is not None:
                paired.append({"name": name, "seed": seed, "bounded_accuracy": a["accuracy"],
                               "uniform_accuracy": b["accuracy"], "difference_percentage_points": 100 * (a["accuracy"] - b["accuracy"])})
    store = EvidenceStore()
    historical = store.read_csv("experiments/outputs/aggregate_bounded_attention_recovery/summary.csv")
    old_best = {r["model"]: float(r["best_accuracy"]) for r in historical}
    for filename, rows in [("per_seed", results), ("summary", summary), ("paired_control", paired)]:
        save_csv(root / f"{filename}.csv", rows)
    payload = {"expected_runs": len(expected), "completed_runs": len(results), "missing": missing,
               "summary": summary, "per_seed": results, "paired_control": paired,
               "interpretation": ["Configured alpha denominator floors are enforced; input-matrix bounds are not certified.",
                                  "Checkpoint selection uses a training-data split; official SST-2 validation is evaluated after selection.",
                                  "Recovery refers to historical one-seed baseline targets minus one percentage point, not statistical equivalence.",
                                  "Uniform controls use the same initial weights, data, optimization budget, and masks.",
                                  "This is short-sentence classification and does not demonstrate long-context content routing."]}
    save_json(root / "summary.json", payload)
    lines = ["BOUNDED RECOVERY EXTENSION", "", f"Completed {len(results)}/{len(expected)} prescribed runs.",
             "Accuracy is measured on the 872 official validation examples after independent checkpoint selection.",
             "Mean +/- sample SD across training seeds; thresholds are historical targets minus 1 percentage point.", "",
             "Model     Mode        Seeds   Accuracy (%)       Threshold (%)   Qualifying seeds"]
    for r in summary:
        deviation = f"{100*r['std_accuracy']:.2f}" if r["std_accuracy"] is not None else "NA"
        lines.append(f"{r['name']:9s} {r['mode']:11s} {r['training_seeds']:5d}   {100*r['mean_accuracy']:.2f} +/- {deviation:5s}     {100*r['historical_recovery_threshold']:.2f}            {r['seeds_meeting_historical_threshold']}")
    lines += ["", "Individual bounded runs:"]
    lines += [f"{r['name']} seed {r['seed']}: {100*r['accuracy']:.2f}%, selected step {r['selected_step']}, threshold met: {r['meets_historical_threshold']}"
              for r in results if r["mode"] == "bounded"]
    lines += ["", "Maximum measured input-matrix norm / configured alpha_s:"]
    lines += [f"{r['name']} {r['mode']} seed {r['seed']}: {r['maximum_matrix_to_alpha_ratio']:.4f}" for r in results]
    lines += ["", "Interpretation:"] + ["- " + s for s in payload["interpretation"]]
    if missing:
        lines += ["", "Missing: " + ", ".join(missing)]
    (root / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for mode, offset, color in [("bounded", -.12, "C0"), ("uniform", .12, "C1")]:
        for i, name in enumerate(names):
            rows = [r for r in results if r["name"] == name and r["mode"] == mode]
            if not rows:
                continue
            acc = np.array([r["accuracy"] * 100 for r in rows])
            axes[0].plot(np.full(len(acc), i + offset), acc, "o", color=color, alpha=.5, markersize=4)
            axes[0].errorbar(i + offset, acc.mean(), yerr=acc.std(ddof=1) if len(acc) > 1 else 0,
                            fmt="s", capsize=4, color=color, label=mode if i == 0 else None)
    for i, name in enumerate(names):
        m = config["models"][name]
        axes[0].plot([i-.3, i+.3], [100*(m["baseline_target"]-.01)]*2, "k--", linewidth=1,
                     label="Historical recovery threshold" if i == 0 else None)
        axes[0].plot(i, 100*old_best[m["model_id"]], marker="x", color="C3", markersize=7,
                     label="Archived bounded best (one seed)" if i == 0 else None)
        values = [r["difference_percentage_points"] for r in paired if r["name"] == name]
        if values:
            axes[1].plot(np.full(len(values), i), values, "o", alpha=.6)
            axes[1].errorbar(i, np.mean(values), yerr=np.std(values, ddof=1) if len(values) > 1 else 0,
                            fmt="ks", capsize=4)
    axes[0].set(ylabel="Official validation accuracy (%)", title="Mean ± SD and individual training seeds")
    axes[0].legend(fontsize=7)
    axes[1].axhline(0, color="grey", linestyle=":")
    axes[1].set(ylabel="Bounded minus uniform accuracy (percentage points)", title="Paired seed differences; mean ± SD")
    for ax in axes:
        ax.set_xticks(range(len(names)), ["TinyStories-"+n.upper() for n in names])
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(root / "recovery.pdf", bbox_inches="tight")
    fig.savefig(root / "recovery.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=(Path(__file__).resolve().parents[1] / 'configs/recovery/recovery_extension.json'))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summarize(args.results, json.loads(args.config.read_text()), args.allow_partial)
