"""Summarize saved experiments and produce standalone scientific plots."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

from experiments.lib.experiment import save_csv, save_json


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def groups(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    return grouped


def numbers(rows, key):
    return np.asarray([float(r[key]) for r in rows])


def bootstrap_mean(values, exponentiate=False):
    values = np.asarray(values)
    rng = np.random.default_rng(2801)
    means = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    if exponentiate:
        means = np.exp(means)
    return np.quantile(means, [0.025, 0.975]).tolist()


def save_figure(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize(root):
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    attention = read_csv(root / "synthetic/attention.csv")
    trials = read_csv(root / "synthetic/sampler_trials.csv")
    selective = read_csv(root / "synthetic/selective_sampler.csv")
    summary = {"scope": "Synthetic comparisons, trained associative recall, and frozen pretrained-model interventions; no quantum implementation or pretrained-model fine-tuning."}
    synthetic_summary = []
    for key, rows in groups(attention, ["N", "case", "scaling"]).items():
        synthetic_summary.append({"N": int(key[0]), "case": key[1], "scaling": key[2], "seeds": len(rows),
                                  **{f"mean_{field}": float(numbers(rows, field).mean()) for field in
                                     ["max_probability_times_keys", "normalized_entropy", "mean_pool_absolute_error", "mean_pool_relative_error"]}})
    sampler_summary = []
    for key, rows in groups(trials, ["N", "case", "scaling", "samples"]).items():
        seed_rmse = [float(np.sqrt(numbers(r, "squared_l2_error").mean())) for r in groups(rows, ["seed"]).values()]
        sampler_summary.append({"N": int(key[0]), "case": key[1], "scaling": key[2], "samples": int(key[3]),
                                "trials": len(rows), "absolute_rmse": float(np.sqrt(numbers(rows, "squared_l2_error").mean())),
                                "seed_rmse_std": float(np.std(seed_rmse, ddof=1)),
                                "success_at_0.01": float(numbers(rows, "success_at_0.01").mean()),
                                "success_at_0.03": float(numbers(rows, "success_at_0.03").mean()),
                                "success_at_0.1": float(numbers(rows, "success_at_0.1").mean())})
    save_csv(root / "synthetic_summary.csv", synthetic_summary)
    save_csv(root / "sampler_summary.csv", sampler_summary)
    summary["synthetic"] = synthetic_summary
    summary["sampler"] = sampler_summary

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for scaling, label in [("theorem_frobenius", "Frobenius scaling"), ("theorem_spectral", "Spectral scaling")]:
        rows = sorted([r for r in synthetic_summary if r["case"] == "exact_copy_identity" and r["scaling"] == scaling], key=lambda x: x["N"])
        axes[0, 0].plot([r["N"] for r in rows], [r["mean_max_probability_times_keys"] for r in rows], "o-", label=label)
        rows = sorted([r for r in synthetic_summary if r["case"] == "random_weights" and r["scaling"] == scaling], key=lambda x: x["N"])
        axes[0, 1].loglog([r["N"] for r in rows], [r["mean_mean_pool_relative_error"] for r in rows], "o-", label=label)
    axes[0, 0].set(xscale="log", ylabel="N × matching-token probability", title="Exact-copy query; unit-norm rows, identity Q/K")
    axes[0, 0].axhline(1, color="grey", linestyle=":")
    axes[0, 1].set(ylabel="Relative L2 error from mean pooling", title="Random unit-spectral-norm weights")
    for count in sorted({r["samples"] for r in sampler_summary}):
        rows = sorted([r for r in sampler_summary if r["case"] == "random_weights" and r["scaling"] == "theorem_frobenius" and r["samples"] == count], key=lambda x: x["N"])
        axes[1, 0].loglog([r["N"] for r in rows], [r["absolute_rmse"] for r in rows], "o-", label=f"T={count}")
    axes[1, 0].set(ylabel="Absolute L2 RMSE", title="Uniform row sampler; 5 instances × 20 draws")
    sel_groups = groups([r for r in selective if int(r["samples"]) == 4000], ["N"])
    ns = sorted(int(n[0]) for n in sel_groups)
    axes[1, 1].semilogx(ns, [numbers(sel_groups[(str(n),)], "empirical_miss_probability").mean() for n in ns], "o", label="Measured")
    axes[1, 1].semilogx(ns, [(1 - 1 / n)**4000 for n in ns], "-", label="(1 − 1/N)^4000")
    axes[1, 1].set(ylabel="Probability of missing the marked token", title="Selective scalar family; logit contrast log(N)", ylim=(-0.03, 1.03))
    for ax in axes.flat:
        ax.set_xlabel("Sequence length N")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    save_figure(fig, figures, "synthetic")

    if (root / "pretrained/manifest.json").exists():
        losses = read_csv(root / "pretrained/losses.csv")
        attn = read_csv(root / "pretrained/attention.csv")
        norm = read_csv(root / "pretrained/normalization.csv")
        residual = read_csv(root / "pretrained/residual.csv")
        weights = read_csv(root / "pretrained/weights.csv")
        noise = read_csv(root / "pretrained/noise.csv")
        quality = []
        for key, rows in groups(losses, ["variant", "N"]).items():
            nll = numbers(rows, "mean_nll")
            low, high = bootstrap_mean(nll, exponentiate=True)
            quality.append({"variant": key[0], "N": int(key[1]), "blocks": len(rows),
                            "scored_tokens": sum(int(r["scored_tokens"]) for r in rows),
                            "mean_nll": float(nll.mean()), "perplexity": math.exp(nll.mean()),
                            "perplexity_block_bootstrap_low": low, "perplexity_block_bootstrap_high": high})
        concentration = []
        for key, rows in groups(attn, ["variant", "N", "attention_type"]).items():
            concentration.append({"variant": key[0], "N": int(key[1]), "attention_type": key[2],
                                  "observations": len(rows), "visible_keys": int(rows[0]["visible_keys"]),
                                  "median_max_probability_times_keys": float(np.median(numbers(rows, "max_probability_times_keys"))),
                                  "p95_max_probability_times_keys": float(np.quantile(numbers(rows, "max_probability_times_keys"), .95)),
                                  "median_normalized_entropy": float(np.median(numbers(rows, "normalized_entropy"))),
                                  "max_logit_range": float(numbers(rows, "logit_range").max())})
        conditioning = []
        for key, rows in groups(norm, ["variant", "N", "normalization"]).items():
            conditioning.append({"variant": key[0], "N": int(key[1]), "normalization": key[2],
                                 "minimum_centered_norm": float(numbers(rows, "centered_norm_min").min()),
                                 "minimum_native_denominator": float(numbers(rows, "native_denominator_min").min()),
                                 "median_of_block_layer_p01_centered_norms": float(np.median(numbers(rows, "centered_norm_p01")))})
        caps = []
        for key, rows in groups(weights, ["variant"]).items():
            caps.append({"variant": key[0], "matrices": len(rows), "active_caps": int(numbers(rows, "cap_active").sum()),
                         "maximum_effective_frobenius_norm": float(numbers(rows, "effective_frobenius_norm").max())})
        noise_summary = []
        for key, rows in groups(noise, ["noise_type", "epsilon_rel", "perturbed_layers"]).items():
            by_block = [numbers(r, "delta_nll").mean() for r in groups(rows, ["block"]).values()]
            low, high = bootstrap_mean(by_block)
            noise_summary.append({"noise_type": key[0], "epsilon_rel": float(key[1]), "perturbed_layers": int(key[2]),
                                  "noise_seeds": len({r["seed"] for r in rows}), "blocks": len(by_block),
                                  "mean_delta_nll": float(np.mean(by_block)),
                                  "delta_nll_block_bootstrap_low": low, "delta_nll_block_bootstrap_high": high,
                                  "perplexity_ratio": math.exp(np.mean(by_block)),
                                  "max_bound_ratio": float(numbers(rows, "max_bound_ratio").max())})
        for name, data in [("quality", quality), ("concentration", concentration), ("conditioning", conditioning), ("caps", caps), ("noise", noise_summary)]:
            summary[name] = data
            save_csv(root / f"{name}_summary.csv", data)
        labels = {"native": "Original", "cap_only": "Caps only", "constrained_native": "Caps + block norms",
                  "constrained_theorem": "Caps + norms + theorem scale", "constrained_uniform": "Caps + norms + uniform"}
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for variant, label in labels.items():
            rows = sorted([r for r in quality if r["variant"] == variant], key=lambda x: x["N"])
            axes[0, 0].plot([r["N"] for r in rows], [r["mean_nll"] for r in rows], "o-", label=label)
            rows = sorted([r for r in concentration if r["variant"] == variant and r["attention_type"] == "global"], key=lambda x: x["N"])
            axes[0, 1].semilogy([r["N"] for r in rows], [r["median_max_probability_times_keys"] for r in rows], "o-", label=label)
        axes[0, 0].set(title="Frozen GPT-Neo 125M; paired WikiText-2 targets", ylabel="Mean negative log likelihood (nats)", xlabel="Context length N")
        axes[0, 1].set(title="Final-query concentration; global layers", ylabel="Median visible-keys × maximum weight", xlabel="Context length N")
        for variant in ["native", "constrained_theorem"]:
            for name in ["ln_1", "ln_2"]:
                rows = sorted([r for r in conditioning if r["variant"] == variant and r["normalization"] == name], key=lambda x: x["N"])
                axes[1, 0].plot([r["N"] for r in rows], [r["minimum_centered_norm"] for r in rows], "o-", label=f"{labels[variant]} {name}")
        axes[1, 0].set(title="Observed minima, not uniform guarantees", ylabel="Minimum pre-LayerNorm centered norm", xlabel="Context length N")
        n = max(r["N"] for r in quality)
        for variant in ["native", "constrained_theorem"]:
            rows = [r for r in residual if r["variant"] == variant and int(r["N"]) == n]
            initial = {r["block"]: float(r["median"]) for r in rows if r["layer"] == "0" and r["boundary"] == "block_input"}
            layers = sorted({int(r["layer"]) for r in rows})
            ratios = [np.mean([float(r["median"]) / initial[r["block"]] for r in rows if int(r["layer"]) == layer and r["boundary"] == "block_output"]) for layer in layers]
            axes[1, 1].plot([i + 1 for i in layers], ratios, "o-", label=labels[variant])
        axes[1, 1].set(title=f"Residual-stream growth at N={n}", ylabel="Mean block-median norm / initial block-median norm", xlabel="Transformer block output")
        for ax in axes.flat:
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
        save_figure(fig, figures, "pretrained")

        fig, ax = plt.subplots(figsize=(8, 5))
        for depth in sorted({r["perturbed_layers"] for r in noise_summary if r["perturbed_layers"]}):
            for kind, style in [("uniform", "o-"), ("rounding", "s--")]:
                rows = sorted([r for r in noise_summary if r["noise_type"] == kind and r["perturbed_layers"] == depth], key=lambda x: x["epsilon_rel"])
                x = [r["epsilon_rel"] for r in rows]
                ax.plot(x, [r["mean_delta_nll"] for r in rows], style, label=f"{kind}, first {depth} blocks")
                if kind == "uniform":
                    ax.fill_between(x, [r["delta_nll_block_bootstrap_low"] for r in rows], [r["delta_nll_block_bootstrap_high"] for r in rows], alpha=.1)
        ax.set(xscale="log", xlabel="Relative coordinate-error bound", ylabel="Mean change in negative log likelihood (nats)",
               title="Pretrained interface sensitivity; bounded-noise surrogate")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        save_figure(fig, figures, "interface_noise")

    if (root / "recall/manifest.json").exists():
        recall_test = read_csv(root / "recall/test.csv")
        recall_training = read_csv(root / "recall/training.csv")
        per_seed = []
        for key, rows in groups(recall_test, ["variant", "N", "seed"]).items():
            count = sum(int(r["examples"]) for r in rows)
            per_seed.append({"variant": key[0], "N": int(key[1]), "seed": int(key[2]),
                             "examples": count, "accuracy": sum(int(r["correct"]) for r in rows) / count,
                             "mean_nll": sum(float(r["mean_nll"]) * int(r["examples"]) for r in rows) / count,
                             "mean_matching_probability": sum(float(r["mean_matching_probability"]) * int(r["examples"]) for r in rows) / count,
                             "minimum_sigma1": min(float(r["minimum_sigma1"]) for r in rows),
                             "minimum_sigma2": min(float(r["minimum_sigma2"]) for r in rows)})
        recall_summary = []
        for key, rows in groups(per_seed, ["variant", "N"]).items():
            recall_summary.append({"variant": key[0], "N": key[1], "training_seeds": len(rows),
                                   "mean_accuracy": float(numbers(rows, "accuracy").mean()),
                                   "std_accuracy": float(numbers(rows, "accuracy").std(ddof=1)),
                                   "mean_matching_probability": float(numbers(rows, "mean_matching_probability").mean()),
                                   "minimum_sigma1": float(numbers(rows, "minimum_sigma1").min()),
                                   "minimum_sigma2": float(numbers(rows, "minimum_sigma2").min())})
        save_csv(root / "recall_per_seed.csv", per_seed)
        save_csv(root / "recall_summary.csv", recall_summary)
        summary["recall"] = recall_summary
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        labels = {"native_sqrt_head_dim": "Standard scale", "theorem_frobenius": "Theorem scale", "uniform": "Uniform attention"}
        for variant, label in labels.items():
            rows = sorted([r for r in recall_summary if r["variant"] == variant], key=lambda x: x["N"])
            ns = [r["N"] for r in rows]
            marker, style = {"native_sqrt_head_dim": ("o", "-"), "theorem_frobenius": ("s", "--"), "uniform": ("x", ":")}[variant]
            axes[0, 0].errorbar(ns, [r["mean_accuracy"] * 100 for r in rows], yerr=[r["std_accuracy"] * 100 for r in rows], marker=marker, linestyle=style, markerfacecolor="none", capsize=3, label=label)
            axes[0, 1].plot(ns, [r["mean_matching_probability"] for r in rows], marker=marker, linestyle=style, markerfacecolor="none", label=label)
            grouped_training = groups([r for r in recall_training if r["variant"] == variant], ["step"])
            steps = sorted(int(k[0]) for k in grouped_training)
            axes[1, 0].plot(steps, [numbers(grouped_training[(str(step),)], "training_accuracy").mean() * 100 for step in steps], label=label)
            axes[1, 1].plot(ns, [r["minimum_sigma1"] for r in rows], "o-", label=f"{label}, first LN")
            axes[1, 1].plot(ns, [r["minimum_sigma2"] for r in rows], "s--", label=f"{label}, second LN")
        axes[0, 0].axhline(25, color="grey", linestyle=":", label="Chance")
        axes[0, 0].set(xscale="log", ylim=(0, 102), xlabel="Test sequence length N", ylabel="Accuracy (%)",
                       title="Held-out recall; mean ± SD over 5 training seeds")
        axes[0, 1].set(xscale="log", yscale="log", xlabel="Test sequence length N", ylabel="Mean probability of matching key", title="Routing under the three attention scales")
        axes[1, 0].set(xlabel="Training step", ylabel="Training accuracy (%)", title="Training at N=64; fixed final checkpoints")
        axes[1, 1].set(xscale="log", xlabel="Test sequence length N", ylabel="Observed minimum centered norm", title="Post-residual normalization inputs")
        for ax in [axes[0, 0], axes[0, 1], axes[1, 1]]:
            ax.set_xticks(ns, [str(n) for n in ns])
            ax.xaxis.set_minor_formatter(NullFormatter())
        for ax in axes.flat:
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        save_figure(fig, figures, "recall")

    save_json(root / "summary.json", summary)
    lines = ["EXPERIMENT RESULTS", "", summary["scope"], "",
             "Synthetic attention and sampler: five independent random instances per N, 20 sampling repeats per instance.",
             "Primary sampler metric is absolute L2 error; relative error is diagnostic only.", ""]
    for row in sampler_summary:
        if row["case"] == "random_weights" and row["scaling"] == "theorem_frobenius" and row["samples"] == 4000:
            lines.append(f"N={row['N']:6d}: sampler absolute RMSE={row['absolute_rmse']:.6g}, success at 0.01={row['success_at_0.01']:.3f}")
    if "quality" in summary:
        lines += ["", "Frozen pretrained context comparison (16 fixed windows; same 128 targets/window at every N):",
                  "variant                     N      mean NLL       perplexity"]
        for row in summary["quality"]:
            lines.append(f"{row['variant']:27s} {row['N']:5d} {row['mean_nll']:13.6f} {row['perplexity']:16.6g}")
        lines += ["", "Cap activity:"]
        lines += [str(row) for row in summary["caps"]]
        lines += ["", "Interface-noise changes (native pretrained model; no adaptation):"]
        for row in summary["noise"]:
            lines.append(f"{row['noise_type']:8s} eps_rel={row['epsilon_rel']:.3g}, first {row['perturbed_layers']:2d} layers: delta NLL={row['mean_delta_nll']:.6f}, perplexity ratio={row['perplexity_ratio']:.6f}")
    if "recall" in summary:
        lines += ["", "Trained associative recall (five training seeds; fixed final checkpoints; independent test examples):",
                  "variant                     N       accuracy mean +/- sample SD"]
        for row in summary["recall"]:
            lines.append(f"{row['variant']:27s} {row['N']:5d}    {100 * row['mean_accuracy']:.2f}% +/- {100 * row['std_accuracy']:.2f} pp")
    lines += ["", "Interpretation limits:",
              "- Recall models train only at N=64; failure in this protocol does not establish impossibility under other training or architectures.",
              "- Frozen interventions do not determine whether fine-tuning can recover quality.",
              "- These pretrained diagnostics do not validate the complete theorem architecture or show a quantum speedup.",
              "- Bootstrap intervals resample fixed token windows, not independent training runs or documents.",
              "- Local attention uses the actual visible-key count; N is not substituted for the mask width.",
              "- Resource tables are explicitly labeled proxies, not concrete oracle counts or hardware estimates.",
              "- The synthetic selective-family experiment checks scalar probabilities and sampling, not a quantum lower-bound proof."]
    (root / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    summarize(parser.parse_args().results)
