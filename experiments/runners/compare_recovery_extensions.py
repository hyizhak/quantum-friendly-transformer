"""Compare both recovery cohorts without selecting only successful settings."""

from __future__ import annotations

import argparse
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


def compare(root, allow_partial):
    cohorts = [("initial", "recovery_extension", "recovery_extension.json"),
               ("lower_lr", "recovery_extension_low_lr", "recovery_extension_low_lr.json"),
               ("replication_3m", "recovery_positive_control", "recovery_positive_control.json")]
    all_rows, per_seed, missing_by_cohort = [], [], {}
    for label, directory, config_file in cohorts:
        if not (root / directory / "summary.json").exists() and allow_partial:
            continue
        summary = json.loads((root / directory / "summary.json").read_text())
        missing_by_cohort[label] = summary['missing']
        config = json.loads((ROOT / "experiments/configs/recovery" / config_file).read_text())
        if summary["missing"] and not allow_partial:
            raise ValueError(f"Incomplete cohort {label}")
        for row in summary["summary"]:
            all_rows.append({"cohort": label, "model_learning_rate": config["models"][row["name"]]["learning_rate"], **row})
        for row in summary["per_seed"]:
            per_seed.append({"cohort": label, **row})
    save_csv(root / "recovery_extension_comparison.csv", all_rows)
    save_csv(root / "recovery_extension_all_seeds.csv", per_seed)
    qualified = [r for r in per_seed if r["mode"] == "bounded" and r["meets_historical_threshold"]]
    bounded = [r for r in per_seed if r["mode"] == "bounded"]
    save_json(root / "recovery_extension_comparison.json", {"summary": all_rows, "per_seed": per_seed,
              "missing_by_cohort": missing_by_cohort,
              "bounded_runs_completed": len(bounded), "bounded_runs_qualifying": len(qualified),
              "qualified_bounded_runs": qualified,
              "models_with_at_least_one_qualifying_new_run": sorted({r["model"] for r in qualified})})
    lines = ["RECOVERY EXTENSION: BOTH LEARNING-RATE SETTINGS AND 3M REPLICATION", "",
             f"Completed {len(bounded)} bounded runs and {len(per_seed)-len(bounded)} uniform-control runs.",
             f"Bounded runs meeting their historical one-percentage-point criterion: {len(qualified)}/{len(bounded)}.",
             "Checkpoint selection uses a training-data split; evaluation uses the 872 official validation examples.", "",
             "Model   Setting          Mode       Accuracy mean +/- sample SD (%)   Qualifying seeds"]
    for row in all_rows:
        deviation = f"{100*row['std_accuracy']:.2f}" if row["std_accuracy"] is not None else "NA"
        lines.append(f"{row['name']:7s} {row['cohort']:16s} {row['mode']:10s} {100*row['mean_accuracy']:.2f} +/- {deviation:5s}                    {row['seeds_meeting_historical_threshold']}/{row['training_seeds']}")
    lines += ["", "Unfinished/canceled runs by cohort: " + json.dumps(missing_by_cohort),
              "Remaining controls for already recovered models were canceled at the user's request; all completed and partial artifacts are retained.",
              "", "Historical recovery thresholds: 1M 81.34%, 3M 80.88%, 8M 82.37%, 33M 83.52%.",
              "The historical native targets are not new matched baselines and this criterion is not an equivalence test.",
              "", "Every qualifying bounded run:"]
    lines += [f"{r['name']} {r['cohort']} seed {r['seed']}: {100*r['accuracy']:.2f}%, step {r['selected_step']}" for r in qualified]
    if not qualified:
        lines.append("None in the completed runs.")
    lines += ["", "Interpretation:",
              "- Both cohorts, all seeds, failures, and uniform controls are retained; no setting is silently discarded.",
              "- The lower-rate cohort is an exploratory follow-up prompted by early results of the initial cohort.",
              "- The 3M cohort repeats a previously successful model; a qualifying repeat does not increase the number of distinct recovered model families.",
              "- The alpha denominator floor is enforced during training and evaluation; the supplied alpha_s values do not certify the whole input-matrix bound.",
              "- Input-matrix norm exceedances are recorded. These results cannot be described as validation of all theorem assumptions.",
              "- Short-sentence classification and recovery against a historical threshold do not establish selective routing, long-context performance, or quantum advantage.",
              "", "Reproduction and limitations: experiments/configs/recovery/RECOVERY_EXTENSION_PROTOCOL.txt.",
              "Per-run configs, source/asset hashes, split indices, predictions, selected checkpoints, and diagnostics are in the cohort directories."]
    (root / "recovery_extension_comparison.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    tex = [r"\begin{table}[t]", r"\centering", r"\begin{tabular}{llccc}", r"\hline",
           r"Model & Setting & Bounded (\%) & Uniform (\%) & Recovery seeds \\", r"\hline"]
    for row in all_rows:
        if row["mode"] != "bounded":
            continue
        controls = [r for r in all_rows if r["name"] == row["name"] and r["cohort"] == row["cohort"] and r["mode"] == "uniform"]
        def formatted(r):
            sd = f"{100*r['std_accuracy']:.2f}" if r["std_accuracy"] is not None else "NA"
            return f"${100*r['mean_accuracy']:.2f} \\pm {sd}$"
        setting = {"initial": "Initial LR", "lower_lr": "Lower LR", "replication_3m": "Replication"}[row["cohort"]]
        uniform = formatted(controls[0]) if controls else "--"
        tex.append(f"TinyStories-{row['name'].upper()} & {setting} & {formatted(row)} & {uniform} & {row['seeds_meeting_historical_threshold']}/{row['training_seeds']} " + r"\\")
    tex += [r"\hline", r"\end{tabular}",
            r"\caption{Additional SST-2 recovery experiments with configured alpha denominator floors. Values are mean and sample standard deviation across completed training seeds. Checkpoints are selected on a split of the training data and then evaluated on the 872 official validation examples. Recovery means accuracy within one percentage point below the corresponding historical baseline; it is not a statistical equivalence claim. Available uniform controls preserve the same masks and training protocol; unfinished controls are listed in the accompanying run ledger. The supplied alpha constants are not certified bounds on the complete attention-input matrices, so these runs do not validate all theorem assumptions. Both learning-rate settings and the separate 3M replication are reported.}",
            r"\label{tab:bounded-recovery-extension}", r"\end{table}"]
    (root / "recovery_extension_table.tex").write_text("\n".join(tex) + "\n")

    names = ["1m", "8m", "33m"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    limits = [100*r["accuracy"] for r in per_seed] + [100*r["historical_recovery_threshold"] for r in per_seed]
    for ax, name in zip(axes, names):
        for i, cohort in enumerate(["initial", "lower_lr"]):
            for mode, offset, color in [("bounded", -.12, "C0"), ("uniform", .12, "C1")]:
                rows = [r for r in per_seed if r["name"] == name and r["cohort"] == cohort and r["mode"] == mode]
                if not rows:
                    continue
                values = np.array([100*r["accuracy"] for r in rows])
                ax.plot(np.full(len(values), i+offset), values, "o", color=color, alpha=.45, markersize=4)
                ax.errorbar(i+offset, values.mean(), yerr=values.std(ddof=1) if len(values)>1 else 0,
                            fmt="s", color=color, capsize=4, label=mode if i==0 else None)
        rows = [r for r in per_seed if r["name"] == name]
        if rows:
            ax.axhline(100*rows[0]["historical_recovery_threshold"], color="black", linestyle="--", linewidth=1,
                       label="Historical threshold")
        ax.set(title="TinyStories-"+name.upper(), xticks=[0,1], xticklabels=["Initial LR", "10× lower LR"],
               xlim=(-.45,1.45), ylim=(min(limits)-2,max(limits)+2))
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Official validation accuracy (%)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Individual seeds and mean ± sample SD; selection split kept separate")
    fig.text(.5, .01, 'n = 3 per group except 1M lower-LR uniform: n = 1; remaining controls canceled.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(root / "recovery_extension_comparison.pdf", bbox_inches="tight")
    fig.savefig(root / "recovery_extension_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    compare(args.results, args.allow_partial)
