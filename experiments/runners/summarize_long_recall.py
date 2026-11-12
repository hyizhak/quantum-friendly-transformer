"""Summarize every fixed-seed long-context cohort; keep scaling regimes separate."""

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
COHORTS = ("long_recall_1024", "long_recall_2048", "long_recall_routing_1024", "long_recall_routing_2048")
LABELS = {"general_alpha1": "General scaling, alpha=1", "strict_frobenius": "Strict Frobenius scaling",
          "native_sqrt_d": "Native sqrt(d) scaling", "uniform": "Mean pooling"}
COLORS = {"general_alpha1": "#1764ab", "strict_frobenius": "#c7542a", "native_sqrt_d": "#278468", "uniform": "#666666"}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_addendum(output, aggregate, per_seed):
    """Generate the table and response from all completed fixed-seed records."""
    lookup = {(r["protocol"], r["train_N"], r["variant"], r["N"]): r for r in aggregate}
    if len(aggregate) != 96 or any(r["training_seeds"] != 3 for r in aggregate):
        return

    def result(protocol, train_n, variant, n):
        r = lookup[protocol, train_n, variant, n]
        return f"{100*r['accuracy_mean']:.2f}\\pm{100*r['accuracy_sd']:.2f}"

    table = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{llrrrr}", r"\hline",
             r"Training & Scaling & \multicolumn{2}{c}{Train through $N=1024$} & \multicolumn{2}{c}{Train through $N=2048$} \\",
             r" & & $N=1024$ & $N=8192$ & $N=2048$ & $N=8192$ \\", r"\hline"]
    names = {"general_alpha1": r"General ($\alpha_0=1$)", "strict_frobenius": r"Strict ($\alpha_0=576N$)",
             "native_sqrt_d": r"Native ($\alpha_0=\sqrt{32}$)", "uniform": "Mean pooling"}
    for protocol in ("classification_only", "routing_auxiliary"):
        for variant in names:
            training = "Labels only" if protocol == "classification_only" else "Labels + index"
            cells = [result(protocol, t, variant, n) for t, n in ((1024, 1024), (1024, 8192), (2048, 2048), (2048, 8192))]
            table.append(training + " & " + names[variant] + " & " + " & ".join("$" + c + "$" for c in cells) + r" \\")
        table.append(r"\hline")
    table += [r"\end{tabular}",
              r"\caption{Long-context paired associative recall, accuracy (\%) as mean $\pm$ sample standard deviation over all three fixed training seeds. The second protocol adds matching-index supervision during training only. Both protocols use a curriculum (500 updates at $N=64$, 500 at $N=256$, 2000 at the stated maximum). Evaluation uses 2048 independently drawn pairs per length; the two members of a pair are dependent. No checkpoint or seed is selected on these tests.}",
              r"\label{tab:long-paired-recall}", r"\end{table}"]
    (output / "comment4_table.tex").write_text("\n".join(table) + "\n")
    auxiliary = [r for r in per_seed if r["protocol"] == "routing_auxiliary" and r["variant"] == "general_alpha1"]
    long_rows = [r for r in auxiliary if r["N"] == 8192]
    sigma1 = min(r["minimum_sigma1"] for r in auxiliary)
    sigma2 = min(r["minimum_sigma2"] for r in auxiliary)
    target_min = min(r["mean_target_probability"] for r in long_rows)
    target_max = max(r["mean_target_probability"] for r in long_rows)
    gaps = [r["mean_pair_hidden_gap"] for r in long_rows]
    noise_min = min(100*r["noise_0.1_accuracy"] for r in long_rows)
    shuffle_min = min(100*r["shuffled_query_accuracy"] for r in long_rows)
    shuffle_max = max(100*r["shuffled_query_accuracy"] for r in long_rows)
    active = sorted({r["active_matrix_caps"] for r in auxiliary})
    max_norm = max(r["maximum_effective_weight_norm"] for r in auxiliary)
    addendum = r"""% Proposed addition to the response to Comment 4. Does not replace response.tex.
\paragraph{Long-context recall and an exact mean-pooling control.}
We added a controlled synthetic retrieval experiment at $d=32$, with separate
curricula ending in 2000 training updates at $N=1024$ or $N=2048$ and held-out
evaluation through $N=8192$ ($N/d=256$). Each input contains $N-2$ random
continuous-key memory tokens, a public marker, and a query copying one memory
key. The target is that memory's binary value. Fresh keys are generated for
every training batch. Counterfactual pairs share keys and queries but reverse
all balanced binary values. Consequently the two inputs have exactly the same
row mean and opposite labels. Any linear value projection followed by uniform
pooling and the same query-dependent downstream computation must give identical
outputs on the pair, and hence exactly $50\%$ accuracy on complete pairs.
This control rules out this mean-pooling explanation of above-chance accuracy;
it does not rule out arbitrary nonlinear per-token pooling architectures.

The model contains one selected-row attention layer, both residual connections,
both LayerNorms, a GELU feed-forward block, and a binary classifier. LayerNorm
uses fixed $\gamma_i=1/\sqrt d$, $\beta_i=0$, and zero additive epsilon. All six
complete matrices ($W_q,W_k,W_v,M_1,M_2$ and the classifier) are learned under
Frobenius caps of 24 enforced on every forward pass. There are no learned
embeddings or attention-output projection in this controlled architecture.
All input rows have norm at most one by construction; float64 checks of every
training and test batch establish $\|S\|_F\le\sqrt N$ for the evaluated inputs.
We explicitly compare $\alpha_0=576N$ (strict Frobenius scaling, $\kappa=1$),
$\alpha_0=1$ (general scaling, $\kappa=576N$), native $\sqrt d$ scaling, and
uniform attention. The general-scaling outcomes do not validate selective
routing under the strict denominator.

For each of two training protocols we ran all four variants at both training
lengths and all three fixed seeds (11, 22, 33), for 48 completed runs. Every run
uses 3000 updates and its final checkpoint. Variants share initialized weights
and training batches within each seed and cohort. Test sets contain 2048 independently generated
pairs (4096 sequences) per length. The test RNG differs from training and is
shared across variants and training seeds. The follow-up protocol uses a new
predetermined test RNG. We report all seeds and sample standard deviations;
there is no selection of seeds or checkpoints on the evaluation set.

The first protocol uses binary classification labels alone. The general-scaling
model reached $100\%$ at the training length for two of three seeds at each
length, while the remaining seed stayed near chance, giving substantial seed
variation. Motivated by the first completed run, we then defined a separately
reported exploratory follow-up adding cross-entropy supervision on the known
matching-memory index, with weight one. This is additional training information,
not an input to inference. All variants receive the same auxiliary objective;
for uniform attention it is the constant $\log N$ and has no routing gradient.
This follow-up must not be described as classification-only training or as a
confirmatory protocol selected without observing earlier results.

"""
    addendum += ("With this auxiliary supervision, general-scaling accuracy is $" + result("routing_auxiliary", 1024, "general_alpha1", 8192)
                 + r"\%$ and $" + result("routing_auxiliary", 2048, "general_alpha1", 8192)
                 + r"\%$ at $N=8192$ for the two training lengths, respectively. "
                 + "Replacing attention by uniform weights gives exactly $50\\%$ for every model and length. "
                 + f"Shuffling query keys across independent pairs gives ${shuffle_min:.2f}$--${shuffle_max:.2f}\\%$ at $N=8192$ across the six auxiliary general-scaling runs. "
                 + f"Their mean attention mass on the matching token is ${target_min:.3f}$--${target_max:.3f}$ (uniform: $1/8192$), and their mean post-LayerNorm counterfactual output separation is ${min(gaps):.3f}$--${max(gaps):.3f}$ in Euclidean norm. "
                 + f"Accuracy remains at least ${noise_min:.2f}\\%$ with random attention-output perturbations of Euclidean norm $0.1$; this is an empirical random-noise diagnostic, not a worst-case robustness guarantee.\n\n")
    addendum += (f"The auxiliary general-scaling models have {','.join(map(str, active))} of six matrix caps active at the final checkpoint (Q and K); "
                 + f"their maximum effective matrix norm is ${max_norm:.4f}<24$. "
                 + f"The minimum observed LayerNorm centered norms across their test lengths and seeds are ${sigma1:.4f}$ and ${sigma2:.4f}$. "
                 + "These are empirical minima, not uniform lower bounds over all possible inputs. Raw/effective norms, cap activity, input bounds, conditioning minima, predictions, training logs, and source/checkpoint hashes are retained.\n\n")
    addendum += r"""The strict Frobenius-scaled models remain approximately at $50\%$, including
with auxiliary supervision. Thus the experiment supports a concrete, trained
separation from mean pooling for the general-scaling model with explicit norm
constraints, while retaining the limitation of the simplified $\kappa=1$
regime. These are fixed-budget comparisons, not an exhaustive optimization of
each scaling. At $N=8192$ the former has $\kappa=4{,}718{,}592$ under the stated
full-matrix bounds; this large factor is distinct from the measured logit
range. Neither the recall scores nor the observed conditioning establish an
efficient quantum implementation or a quantum--classical speedup. These
synthetic single-layer results also do not establish natural-language quality.
See Table~\ref{tab:long-paired-recall} and the complete per-seed results.
"""
    (output / "comment4_addendum.tex").write_text(addendum)


def summarize(root, allow_partial=False):
    output = root / "long_recall_summary"
    output.mkdir(exist_ok=True)
    per_seed, pending = [], []
    for cohort in COHORTS:
        directory = root / cohort
        if not (directory / "plan.json").exists():
            pending.append(cohort)
            continue
        plan = json.loads((directory / "plan.json").read_text())
        for item in plan:
            run = directory / f"{item['variant']}_seed{item['seed']}"
            if item["status"] != "completed" or not (run / "manifest.json").exists():
                pending.append(str(run.relative_to(root)))
                continue
            result = json.loads((run / "result.json").read_text())
            assert not result["smoke"]
            metrics = read_csv(run / "test.csv")
            weights = read_csv(run / "weights_evaluation_float64.csv")
            for evaluation in result["evaluation"]:
                n = evaluation["N"]
                batches = [r for r in metrics if int(r["N"]) == n]
                examples = sum(int(r["examples"]) for r in batches)
                average = lambda key: sum(float(r[key]) * int(r["examples"]) for r in batches) / examples
                per_seed.append({
                    "cohort": cohort, "protocol": "routing_auxiliary" if "routing" in cohort else "classification_only",
                    "train_N": result["training_maximum_N"], "variant": result["variant"], "seed": result["seed"],
                    **evaluation, "mean_nll": average("mean_nll"), "mean_margin": average("mean_margin"),
                    "mean_target_probability": average("mean_target_probability"),
                    "mean_pair_hidden_gap": average("mean_pair_hidden_gap"),
                    "mean_normalized_entropy": average("mean_normalized_entropy"),
                    "minimum_sigma1": min(float(r["minimum_sigma1"]) for r in batches),
                    "minimum_sigma2": min(float(r["minimum_sigma2"]) for r in batches),
                    "maximum_absolute_logit": max(float(r["maximum_absolute_logit"]) for r in batches),
                    "maximum_logit_contrast": max(float(r["maximum_logit_contrast"]) for r in batches),
                    "kappa": float(batches[0]["kappa"]),
                    "maximum_effective_weight_norm": max(float(r["effective_frobenius"]) for r in weights),
                    "active_matrix_caps": sum(r["cap_active"] == "True" for r in weights),
                    "result_path": str(run.relative_to(ROOT) / "result.json"),
                })
    if pending and not allow_partial:
        raise RuntimeError(f"Incomplete cohorts: {pending}")
    if not per_seed:
        raise RuntimeError("No completed results")
    groups = {}
    for row in per_seed:
        groups.setdefault((row["protocol"], row["train_N"], row["variant"], row["N"]), []).append(row)
    aggregate = []
    measures = ["accuracy", "uniform_accuracy", "shuffled_query_accuracy", "noise_0.0001_accuracy", "noise_0.01_accuracy", "noise_0.1_accuracy",
                "mean_target_probability", "mean_pair_hidden_gap", "mean_margin", "mean_normalized_entropy"]
    for (protocol, train_n, variant, n), rows in sorted(groups.items()):
        record = {"protocol": protocol, "train_N": train_n, "variant": variant, "N": n, "training_seeds": len(rows)}
        for key in measures:
            values = [r[key] for r in rows]
            record[key + "_mean"] = statistics.mean(values)
            record[key + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        record.update(minimum_sigma1=min(r["minimum_sigma1"] for r in rows), minimum_sigma2=min(r["minimum_sigma2"] for r in rows),
                      maximum_effective_weight_norm=max(r["maximum_effective_weight_norm"] for r in rows),
                      maximum_kappa=max(r["kappa"] for r in rows))
        aggregate.append(record)
    write_addendum(output, aggregate, per_seed)
    save_csv(output / "per_seed.csv", per_seed)
    save_csv(output / "aggregate.csv", aggregate)
    (output / "summary.json").write_text(json.dumps({"completed_models": len(per_seed) // 6,
        "pending": pending, "cohorts": COHORTS, "aggregates": aggregate,
        "interpretation": "Strict kappa=1 and general-scaling results are separate; auxiliary supervision is disclosed; no quantum speedup is measured."}, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, constrained_layout=True)
    for i, protocol in enumerate(("classification_only", "routing_auxiliary")):
        for j, train_n in enumerate((1024, 2048)):
            ax = axes[i, j]
            for variant in LABELS:
                rows = sorted([r for r in aggregate if (r["protocol"], r["train_N"], r["variant"]) == (protocol, train_n, variant)], key=lambda r: r["N"])
                if not rows:
                    continue
                x, y = [r["N"] for r in rows], [100 * r["accuracy_mean"] for r in rows]
                sd = [100 * (r["accuracy_sd"] or 0) for r in rows]
                ax.plot(x, y, marker="o", markersize=3, linestyle="--" if variant == "uniform" else "-", label=LABELS[variant], color=COLORS[variant])
                if variant != "uniform":
                    ax.fill_between(x, [max(0, a-b) for a,b in zip(y,sd)], [min(100, a+b) for a,b in zip(y,sd)], alpha=.12, color=COLORS[variant])
            ax.axvline(train_n, color="#bbbbbb", linestyle=":", linewidth=1)
            ax.set_xscale("log", base=2)
            ax.set_xticks([256, 512, 1024, 2048, 4096, 8192], ["256", "512", "1024", "2048", "4096", "8192"])
            ax.set_ylim(45, 102)
            ax.grid(alpha=.15)
            title = "Classification labels" if i == 0 else "Classification + matching-index supervision"
            ax.set_title(f"{title}\nFinal training length {train_n}", fontsize=10)
            if i == 1: ax.set_xlabel("Evaluation sequence length N")
            if j == 0: ax.set_ylabel("Accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    fig.savefig(output / "accuracy.pdf")
    fig.savefig(output / "accuracy.png", dpi=170)
    plt.close(fig)
    (output / "figure_captions.txt").write_text(
        "accuracy: Mean test accuracy across all three fixed training seeds. Shading is one sample standard deviation, clipped to [0,100]. "
        "Columns use final training lengths 1024 and 2048; rows show classification-only and the separately reported matching-index auxiliary protocol. "
        "Dotted vertical lines mark maximum training length. Test sets contain 2048 independent counterfactual pairs per length.\n\n"
        "routing_diagnostics: Matching-token attention probability and post-LayerNorm output distance between counterfactual partners, "
        "for the auxiliary protocol trained through N2048. Means and one sample SD across all three seeds. "
        "The pairs have identical mean-pooled inputs and opposite answers; uniform outputs are identical.\n")

    diagnostic_rows = [r for r in aggregate if r["protocol"] == "routing_auxiliary" and r["train_N"] == 2048]
    if diagnostic_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
        for variant in LABELS:
            rows = sorted([r for r in diagnostic_rows if r["variant"] == variant], key=lambda r: r["N"])
            if not rows:
                continue
            x = [r["N"] for r in rows]
            for ax, key in zip(axes, ("mean_target_probability", "mean_pair_hidden_gap")):
                y = [r[key + "_mean"] for r in rows]
                sd = [r[key + "_sd"] or 0 for r in rows]
                ax.plot(x, y, marker="o", markersize=3, color=COLORS[variant], label=LABELS[variant],
                        linestyle="--" if variant == "uniform" else "-")
                ax.fill_between(x, [max(0, a-b) for a,b in zip(y,sd)], [a+b for a,b in zip(y,sd)], alpha=.12, color=COLORS[variant])
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xticks([256, 512, 1024, 2048, 4096, 8192], ["256", "512", "1024", "2048", "4096", "8192"])
            ax.set_xlabel("Evaluation sequence length N")
            ax.grid(alpha=.15)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Attention probability on matching token")
        axes[1].set_ylabel("Post-LayerNorm paired output L2 distance")
        axes[1].set_ylim(-.05, 2.05)
        fig.suptitle("Matching-index auxiliary training through N=2048; all three seeds", fontsize=11)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
        fig.savefig(output / "routing_diagnostics.pdf")
        fig.savefig(output / "routing_diagnostics.png", dpi=170)
        plt.close(fig)

    lines = ["LONG-CONTEXT RECALL WITH IDENTICAL MEAN-POOLED COUNTERFACTUAL INPUTS", "",
             f"Completed models: {len(per_seed)//6}; pending entries: {len(pending)}.",
             "Three seeds per complete setting; means and sample SD below. Every seed is retained.",
             "Curriculum: 500 updates at N64, 500 at N256, 2000 at the stated long length.",
             "Evaluation: 2048 independent pairs / 4096 sequences per length, fixed final checkpoint.",
             "Original and auxiliary protocols use distinct predetermined test seeds.", ""]
    for row in aggregate:
        if row["N"] not in (row["train_N"], 8192):
            continue
        sd = 100 * row["accuracy_sd"] if row["accuracy_sd"] is not None else float("nan")
        lines.append(f"{row['protocol']:20s} trainN={row['train_N']:4d} {row['variant']:17s} evalN={row['N']:4d}: "
                     f"{100*row['accuracy_mean']:.2f} +/- {sd:.2f}%; "
                     f"uniform ablation={100*row['uniform_accuracy_mean']:.2f}%, "
                     f"shuffled query={100*row['shuffled_query_accuracy_mean']:.2f}%, "
                     f"noise L2 .01={100*row['noise_0.01_accuracy_mean']:.2f}%")
    lines += ["", "Interpretation:",
              "- The paired data provide an exact 50% ceiling for this mean-pooling architecture, regardless of its optimizer.",
              "- Only strict_frobenius uses alpha0=N*24^2 and kappa=1. General-alpha1 has kappa=N*24^2.",
              "- Auxiliary training adds the known matching-index loss; that index never enters inference.",
              "- All models use unit-bounded inputs, capped complete matrices, two residuals and fixed unit-norm LayerNorms.",
              "- Positive general-scaling accuracy is not evidence of selective routing under the strict Frobenius denominator.",
              "- This synthetic task does not establish pretrained language-model quality or quantum computational advantage."]
    (output / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/reviewer")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summarize(args.results.resolve(), args.allow_partial)
