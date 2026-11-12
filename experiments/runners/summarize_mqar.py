"""Summarize complete MQAR cohorts, preserving failed initialization attempts."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.lib.reviewer import save_csv, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reviewer/mqar_summary")
    parser.add_argument("--initial-results", type=Path, default=ROOT / "results/reviewer/mqar")
    parser.add_argument("--followup-results", type=Path, default=ROOT / "results/reviewer/mqar_variance_init")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cohorts = [("initial", args.initial_results, "Default initialization"),
               ("variance_init", args.followup_results, "Initialization follow-up")]
    all_rows, curves, audits, norms = [], [], {}, []
    for cohort, root, label in cohorts:
        for seed in (11,22,33):
            for variant in ("native", "uniform"):
                folder = root / f"{variant}_seed{seed}"
                result = json.loads((folder / "result.json").read_text())
                if result["smoke"]:
                    raise AssertionError("Smoke result cannot enter benchmark summary")
                for row in result["evaluation"]:
                    all_rows.append({"cohort":cohort, "variant":variant, "seed":seed,
                        **{k:v for k,v in row.items() if k != "layer_diagnostics"},
                        "sigma1_min":min(r["sigma1_min"] for r in row["layer_diagnostics"]),
                        "sigma2_min":min(r["sigma2_min"] for r in row["layer_diagnostics"]),
                        "run":str(folder)})
                for row in json.loads((folder / "validation.json").read_text()):
                    curves.append({"cohort":cohort,"variant":variant,"seed":seed,
                                   "epoch":row["epoch"],"accuracy":row["accuracy"],"mean_nll":row["mean_nll"]})
                with (folder / "weights_final.csv").open() as stream:
                    weights = list(csv.DictReader(stream))
                capped = [r for r in weights if r["capped"] == "True"]
                norms.append({"cohort":cohort,"variant":variant,"seed":seed,
                              "capped_matrices":len(capped),"active_caps":sum(r["cap_active"] == "True" for r in capped),
                              "maximum_effective_norm":max(float(r["effective_frobenius"]) for r in capped)})
        audit_path = root / "audit.json"
        audits[cohort] = json.loads(audit_path.read_text()) if audit_path.exists() else {"status":"pending"}
    aggregates = []
    for key in sorted({(r["cohort"],r["variant"],r["N"],r["pairs"]) for r in all_rows}):
        rows = [r for r in all_rows if (r["cohort"],r["variant"],r["N"],r["pairs"]) == key]
        if len(rows) != 3 or {r["seed"] for r in rows} != {11,22,33}:
            raise AssertionError("Incomplete seed cohort")
        record = dict(zip(("cohort","variant","N","pairs"),key))
        record.update(seeds=3,role=rows[0]["role"],queries_per_seed=rows[0]["queries"])
        for metric in ("accuracy","mean_nll","sequence_accuracy","uniform_ablation_accuracy"):
            record[metric+"_mean"] = statistics.mean(r[metric] for r in rows)
            record[metric+"_sd"] = statistics.stdev(r[metric] for r in rows)
        aggregates.append(record)
    save_csv(args.output / "per_seed.csv",all_rows)
    save_csv(args.output / "aggregate.csv",aggregates)
    save_csv(args.output / "validation_curves.csv",curves)
    save_csv(args.output / "cap_activity.csv",norms)
    save_json(args.output / "summary.json",{"models":12,"all_seeds_retained":True,
        "benchmark":"Official Zoology MQAR generator and full-vocabulary query-token accuracy",
        "scope":"Selected standard condition with an initialization follow-up; not a full published sweep reproduction",
        "aggregates":aggregates,"cap_activity":norms,"audits":audits})
    lines = ["MQAR BENCHMARK EVALUATION", "",
        "Official source: https://github.com/HazyResearch/zoology/tree/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb",
        "Paper: https://arxiv.org/abs/2312.04927", "",
        "Vocabulary 8,192, random filler tokens, power_a=.01, one context pass.",
        "Primary condition: N=256, 16 key-value pairs; 20,000 train, 1,000 validation, 1,000 test sequences.",
        "Answer-token supervision only. Full-vocabulary argmax at every official nonignored query.",
        "Two layers, d=64, FFN width 128, cap 64, native attention sqrt(64), unit-norm LayerNorms.",
        "All 12 Transformer projections capped; token inputs row-normalized; vocabulary readout uncapped.",
        "32 full epochs, 5,024 updates, paired initialization/data order; fixed final checkpoints; all 3 seeds.",
        "Initial cohort retained in full. Initialization follow-up motivated by validation/training failure",
        "and evaluated on a fresh test stream. It is exploratory, not an unselected confirmatory protocol.",
        "Values below are mean +/- sample SD over training seeds. Queries within a sequence are dependent.", ""]
    for cohort, root, label in cohorts:
        lines += [label]
        for r in [r for r in aggregates if r["cohort"] == cohort]:
            lines.append(f"  {r['variant']:7s} N={r['N']:4d} K={r['pairs']:3d}: {100*r['accuracy_mean']:.3f} +/- {100*r['accuracy_sd']:.3f}%  uniform substitution={100*r['uniform_ablation_accuracy_mean']:.3f}%  ({r['role']})")
        lines.append(f"  Audit: {audits[cohort]['status']}")
        primary = [r for r in all_rows if r["cohort"] == cohort and r["variant"] == "native" and (r["N"],r["pairs"]) == (256,16)]
        lines.append("  Primary native results by seed: " + ", ".join(f"{r['seed']}={100*r['accuracy']:.5f}%" for r in primary))
        for r in [r for r in norms if r["cohort"] == cohort]:
            lines.append(f"  {r['variant']} seed{r['seed']}: active final caps {r['active_caps']}/{r['capped_matrices']}; max effective projection norm {r['maximum_effective_norm']:.4f}")
        lines.append("")
    lines += ["Interpretation limits:",
        "These are classical benchmark scores for the specified trained model, not pretrained-language quality",
        "or computational/quantum advantage. Cap 64 must not be called FN24. Longer lengths and increased",
        "memory density differ from the training distribution. Failure to train under one recipe does not",
        "prove an architectural impossibility. No test-dependent checkpoint or seed selection was performed.", ""]
    (args.output / "summary.txt").write_text("\n".join(lines))
    conditions = [(64,4),(128,8),(256,16),(256,64),(512,32),(1024,64),(2048,128)]
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),sharey=True,layout="constrained")
    for ax,(cohort,_,label) in zip(axes,cohorts):
        for variant,color,title in [("native","#245c9f","Native attention"),("uniform","#b05a32","Trained uniform control")]:
            rows = [next(r for r in aggregates if (r["cohort"],r["variant"],r["N"],r["pairs"]) == (cohort,variant,n,k)) for n,k in conditions]
            means = np.array([r["accuracy_mean"] for r in rows])*100
            ax.plot(range(len(rows)),means,marker="o",color=color,label=title)
            for index,seed in enumerate((11,22,33)):
                values = [100*next(r["accuracy"] for r in all_rows if (r["cohort"],r["variant"],r["seed"],r["N"],r["pairs"]) == (cohort,variant,seed,n,k)) for n,k in conditions]
                ax.scatter(np.arange(len(rows))+(index-1)*.06,values,color=color,alpha=.6,s=16,zorder=3)
        ax.set_xticks(range(len(conditions)),[f"{n}\n{k} KV" for n,k in conditions])
        ax.set_xlabel("Sequence length and memory pairs")
        ax.set_title(label)
        ax.set_ylim(-2,102)
        ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Official MQAR query-token accuracy (%)")
    axes[1].legend(loc="best",fontsize=8)
    fig.suptitle("MQAR: lines show means; dots show all three seeds; cap 64")
    fig.savefig(args.output / "mqar_accuracy.pdf")
    fig.savefig(args.output / "mqar_accuracy.png",dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4),sharey=True,layout="constrained")
    for ax,(cohort,_,label) in zip(axes,cohorts):
        for variant,color in [("native","#245c9f"),("uniform","#b05a32")]:
            for seed in (11,22,33):
                rows = [r for r in curves if (r["cohort"],r["variant"],r["seed"]) == (cohort,variant,seed)]
                ax.plot([r["epoch"] for r in rows],[100*r["accuracy"] for r in rows],color=color,alpha=.7,label=variant if seed == 11 else None)
        ax.set_title(label)
        ax.set_xlabel("Epoch (validation diagnostics; no checkpoint selection)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Validation query-token accuracy (%)")
    axes[1].legend()
    fig.savefig(args.output / "mqar_validation.pdf")
    plt.close(fig)
    latex = [r"\begin{table}[t]",r"\centering\small",r"\begin{tabular}{llrrrr}",r"\hline",
             r"Initialization & Attention & $N=256,K=16$ & $N=512,K=32$ & $N=1024,K=64$ & $N=2048,K=128$ \\",r"\hline"]
    for cohort,_,label in cohorts:
        for variant in ("native","uniform"):
            cells=[]
            for n,k in [(256,16),(512,32),(1024,64),(2048,128)]:
                r=next(r for r in aggregates if (r["cohort"],r["variant"],r["N"],r["pairs"]) == (cohort,variant,n,k))
                cells.append(f"${100*r['accuracy_mean']:.3f}\\pm{100*r['accuracy_sd']:.3f}$")
            latex.append(f"{label} & {variant} & " + " & ".join(cells) + r" \\")
    latex += [r"\hline",r"\end{tabular}",
        r"\caption{Official-generator MQAR query-token accuracy (\%), mean $\pm$ sample SD across all three seeds. Vocabulary 8192; training uses 20000 sequences at $N=256,K=16$, 32 epochs, and answer labels only. All complete Transformer projections have cap 64; native temperature is $\sqrt{64}$. The vocabulary readout is uncapped. The initialization follow-up is exploratory and uses a fresh test stream. Longer contexts also contain more memory pairs and are outside the training distribution. Both complete cohorts are retained.}",
        r"\label{tab:mqar-benchmark}",r"\end{table}"]
    (args.output / "mqar_table.tex").write_text("\n".join(latex)+"\n")
    def percentage(cohort,variant,n,k):
        r=next(r for r in aggregates if (r["cohort"],r["variant"],r["N"],r["pairs"]) == (cohort,variant,n,k))
        return f"${100*r['accuracy_mean']:.3f}\\pm{100*r['accuracy_sd']:.3f}\\%$"
    paragraph = r"""\textbf{Established associative-recall benchmark.}
We additionally evaluated the Multi-Query Associative Recall (MQAR) benchmark
using the official Zoology generator pinned to commit
\texttt{1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb}. The vocabulary contains 8192
tokens; random non-query tokens and the original power-law query placement are
retained. Accuracy is measured at every official labeled query over the full
vocabulary. The primary condition uses $N=256$ and 16 key--value pairs, with 20000
training sequences and separate 1000-sequence validation and test sets. This is
a selected standard condition, not a reproduction of the complete published sweep.

The two-layer model has $d=64$, native temperature $\sqrt{64}$, both residuals,
fixed unit-norm LayerNorms, and Frobenius caps of 64 on all 12 Transformer
projection matrices. Token inputs are row-normalized; the separate vocabulary
readout is uncapped. We train all three seeds for 32 epochs using answer-token
labels only, with no matching-index supervision, and evaluate fixed final
checkpoints. Separately trained causal uniform-attention controls use paired
initial weights and training orders. There is no claimed exact 50\% ceiling for
this multilayer MQAR control.

The original initialization was unstable across seeds. We retain that complete
cohort and report an exploratory follow-up that adjusts projection initialization
for unit-norm activations while retaining the architecture, caps, attention
temperature, data and optimization budget. This adjustment was motivated by
training/validation failure; the follow-up uses a fresh test stream. It is not
an unselected confirmatory training protocol.

"""
    paragraph += f"The original native cohort scores {percentage('initial','native',256,16)} on the primary condition. With the adjusted initialization, native attention achieves {percentage('variance_init','native',256,16)}, compared with {percentage('variance_init','uniform',256,16)} for the trained uniform control (mean and sample standard deviation across all three seeds). "
    paragraph += f"At $N=2048$ with 128 memory pairs, native accuracy falls to {percentage('variance_init','native',2048,128)}; sequence length and memory load both increase relative to training. "
    paragraph += "The results establish successful retrieval for the primary benchmark configuration, with limited generalization to the harder conditions. They are not FN24 recovery results or evidence of computational advantage. All seeds, the original cohort, per-query predictions, cap diagnostics and replay checks are retained.\n\n"
    paragraph += r"\input{results/reviewer/mqar_summary/mqar_table.tex}" + "\n"
    (args.output / "comment4_mqar_insert.tex").write_text(paragraph)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
