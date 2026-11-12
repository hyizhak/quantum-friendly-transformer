"""Report every validation recipe and the reserved-test N8192 comparison."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--validation-audit', type=Path, required=True)
    parser.add_argument('--test-audit', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selection = json.loads(args.selection.read_text())
    validation_audit = json.loads(args.validation_audit.read_text())
    test_audit = json.loads(args.test_audit.read_text())
    assert validation_audit["status"] == test_audit["status"] == "passed"
    final_root = args.results
    groups = [("selected_cap24", "Selected cap24", "native_sqrt_d"),
              ("reference_cap64", "Cap64 reference", "native_sqrt_d"),
              ("uniform_cap24", "Uniform cap24", "uniform")]
    per_seed, hashes = [], {}
    for group, label, variant in groups:
        for seed in (11, 22, 33):
            run = final_root / group / f"{variant}_seed{seed}"
            path = run / "result.json"
            result = json.loads(path.read_text())
            config = json.loads((run / "config.json").read_text())
            hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
            for row in result["evaluation"]:
                per_seed.append({"group": group, "label": label, "seed": seed,
                                 "cap": config["cap"], "training_steps": config["steps"],
                                 "routing_weight": config["routing_auxiliary_weight"], **row})
    aggregates = []
    for group, label, _ in groups:
        for n in sorted({r["N"] for r in per_seed}):
            rows = [r for r in per_seed if r["group"] == group and r["N"] == n]
            assert len(rows) == 3
            aggregate = {"group": group, "label": label, "N": n, "seeds": 3}
            for metric in ["accuracy", "uniform_accuracy", "shuffled_query_accuracy",
                           "noise_0.0001_accuracy", "noise_0.01_accuracy", "noise_0.1_accuracy"]:
                aggregate[metric+"_mean"] = statistics.mean(r[metric] for r in rows)
                aggregate[metric+"_sd"] = statistics.stdev(r[metric] for r in rows)
            aggregates.append(aggregate)
    lookup = {(r["group"], r["N"]): r for r in aggregates}
    selected, reference = lookup["selected_cap24", 8192], lookup["reference_cap64", 8192]
    gap = 100 * (reference["accuracy_mean"] - selected["accuracy_mean"])
    verdict = ("The selected cap24 recipe did not recover the cap64 reference performance."
               if gap > 0 else "The selected cap24 recipe matched or exceeded reference accuracy on this test.")
    summary = {"task": "Continuous-key balanced single-query recall; not MQAR",
               "selection": selection, "evaluation_role": "reserved_final_test",
               "test_seed": 7041937, "pairs_per_length": 2048,
               "cap24_minus_reference_gap_percentage_points": -gap,
               "verdict": verdict, "aggregates": aggregates,
               "validation_audit": validation_audit, "final_test_audit": test_audit,
               "source_result_sha256": hashes}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(args.output / "per_seed.csv", per_seed)
    write_csv(args.output / "aggregate.csv", aggregates)
    write_csv(args.output / "validation_recipes.csv", selection["recipes"])
    def score(row):
        return f"{100*row['accuracy_mean']:.3f} +/- {100*row['accuracy_sd']:.3f}%"
    lines = ["CAP24 TRAINING THROUGH N=8192", "",
             "Frozen continuous-key single-query task; d32; all six matrices capped; native sqrt(32) scaling.",
             "Not MQAR and not pretrained SST-2 recovery. Input and LayerNorm constraints retained.",
             "Every recipe retains all three seeds11,22,33. Fixed final checkpoints.",
             "Validation recipe selection occurred before opening the independent final test.",
             f"Selected recipe: {selection['selected_recipe']}",
             f"Final N8192 cap24: {score(selected)}",
             f"Final N8192 cap64: {score(reference)}",
             f"Gap below cap64: {gap:.3f} percentage points.", verdict, "",
             "All validation recipes (mean +/- sample SD at N8192):"]
    for row in selection["recipes"]:
        lines.append(f"  {row['recipe']}: {row['validation_mean_pct']:.3f} +/- {row['validation_sd_pct']:.3f}%; "
                     f"total training updates={row['total_updates']}; auxiliary weight={row['routing_weight']}")
    lines += ["", "Final-test curves (all three seeds):"]
    for n in sorted({r["N"] for r in aggregates}):
        lines.append(f"N={n}: " + "; ".join(f"{label}: {score(lookup[group,n])}" for group,label,_ in groups))
    lines += ["", "Constraints and interpretation:",
              "Warm recipes load a trained cap64 parent, apply maximum24 before inference, and fine-tune under that cap.",
              "The parent's3000 training updates are an extra resource and are counted in the recipe ledger.",
              "The original8192 scores from models trained through1024/2048 remain separate historical results.",
              "All stored predictions audited; fresh first-batch inference replays at every length; not a second full test.",
              "A failure of these recipes is not a proof that no cap24 architecture or training procedure can succeed."]
    (args.output / "summary.txt").write_text("\n".join(lines) + "\n")
    table = [r"\begin{table}[tbp]", r"\centering\small", r"\begin{tabular}{rrrr}", r"\hline",
             r"Test $N$ & Selected cap 24 & Cap 64 reference & Uniform cap 24 \\", r"\hline"]
    for n in sorted({r["N"] for r in aggregates}):
        values = [f"${100*lookup[group,n]['accuracy_mean']:.3f}\\pm{100*lookup[group,n]['accuracy_sd']:.3f}$"
                  for group,_,_ in groups]
        table.append(f"{n} & " + " & ".join(values) + r" \\")
    table += [r"\hline", r"\end{tabular}",
              r"\caption{Additional single-query recall experiment with training through $N=8192$, separate from MQAR. Accuracy (\%), mean $\pm$ sample standard deviation over all three seeds, on the reserved final test (2048 independent counterfactual pairs per length). The cap-24 recipe is selected by validation accuracy; its training history and additional resources are disclosed in the text. Every projection, including the classifier, remains capped, and native attention scaling is unchanged.}",
              r"\label{tab:cap24-train8192}", r"\end{table}"]
    (args.output / "results_table.tex").write_text("\n".join(table) + "\n")
    fig, ax = plt.subplots(figsize=(8,4.5), constrained_layout=True)
    for group,label,_ in groups:
        rows = [r for r in aggregates if r["group"] == group]
        xs=[r["N"] for r in rows]
        ys=[100*r["accuracy_mean"] for r in rows]
        sd=[100*r["accuracy_sd"] for r in rows]
        ax.errorbar(xs,ys,yerr=sd,marker="o",label=label,capsize=3)
    ax.set_xscale("log",base=2)
    ax.set_xticks([256,512,1024,2048,4096,8192],['256','512','1024','2048','4096','8192'])
    ax.set(xlabel="Evaluation sequence length",ylabel="Accuracy (%)",ylim=(45,102),
           title="Single-query recall: training through 8192, reserved final test")
    ax.grid(alpha=.2);ax.legend()
    for suffix in ('pdf','png'):
        fig.savefig(args.output / f"recall8192.{suffix}",dpi=180)
    plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
