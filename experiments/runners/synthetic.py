"""Measure diffuse attention, uniform-row sampling, and a selective scalar family."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.lib.experiment import (attention_statistics, manifest, sampled_attention,
                                     save_csv, save_json, softmax, taylor_degree)


def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    measurements, errors, selective, resources = [], [], [], []
    d = config["dimension"]
    started = time.perf_counter()
    for seed in config["seeds"]:
        for n in config["sequence_lengths"]:
            rng = np.random.default_rng(np.random.SeedSequence([seed, n]))
            rows = rng.standard_normal((n, d))
            rows /= np.linalg.norm(rows, axis=1, keepdims=True)
            weights = []
            for _ in range(3):
                w = rng.standard_normal((d, d))
                weights.append(w / np.linalg.norm(w, ord=2))
            for case in ["random_weights", "exact_copy_identity"]:
                s = rows.copy()
                wq, wk, wv = weights if case == "random_weights" else [np.eye(d)] * 3
                if case == "exact_copy_identity":
                    s[0] = s[-1]
                fro_squared = float(np.sum(s * s))
                spectral_squared = float(np.linalg.eigvalsh(s.T @ s)[-1])
                mean_output = s.mean(axis=0) @ wv
                coefficient = wk @ (s[-1] @ wq)
                for mode, denominator in [("native_sqrt_head_dim", math.sqrt(d)),
                                          ("theorem_frobenius", fro_squared),
                                          ("theorem_spectral", spectral_squared)]:
                    c = coefficient / denominator
                    logits = s @ c
                    exact = (softmax(logits) @ s) @ wv
                    exact_norm = float(np.linalg.norm(exact))
                    mean_error = float(np.linalg.norm(exact - mean_output))
                    base = {"seed": seed, "N": n, "dimension": d, "case": case, "scaling": mode}
                    measurements.append({**base, **attention_statistics(logits),
                                         "score_denominator": denominator,
                                         "alpha_s_frobenius": math.sqrt(fro_squared),
                                         "alpha_s_spectral": math.sqrt(spectral_squared),
                                         "attention_output_norm": exact_norm,
                                         "mean_pool_absolute_error": mean_error,
                                         "mean_pool_relative_error": mean_error / max(exact_norm, 1e-30),
                                         "copy_token_probability": float(softmax(logits)[0]) if case == "exact_copy_identity" else ""})
                    if mode == "theorem_frobenius" and np.abs(logits).max() > 1 / n + 1e-12:
                        raise AssertionError("Frobenius logit bound violated")
                    for count in config["sample_counts"]:
                        for repeat in range(config["sampler_repeats"]):
                            # Pair the sampled indices across attention modes and matrix cases.
                            sample_rng = np.random.default_rng(np.random.SeedSequence([seed, n, count, repeat, 123]))
                            indices = sample_rng.integers(0, n, size=count)
                            estimate = sampled_attention(s, c, wv, indices)
                            error = float(np.linalg.norm(estimate - exact))
                            errors.append({**base, "samples": count, "repeat": repeat,
                                           "absolute_l2_error": error, "squared_l2_error": error**2,
                                           "relative_l2_error": error / max(exact_norm, 1e-30),
                                           **{f"success_at_{eps:g}": int(error <= eps) for eps in config["absolute_error_targets"]}})
            print(f"completed synthetic seed={seed}, N={n}", flush=True)
    for seed in config["seeds"]:
        for n in config["sequence_lengths"]:
            # This checks the scalar softmax family in response.tex, not its unprovided full-layer proof.
            exact = 1 / (2 * n - 1)
            for count in config["sample_counts"]:
                rng = np.random.default_rng(np.random.SeedSequence([seed, n, count, 456]))
                hits = rng.binomial(count, 1 / n, size=config["selective_repeats"])
                estimates = (n * hits - (count - hits)) / (n * hits + (count - hits))
                error = np.abs(estimates - exact)
                selective.append({"seed": seed, "N": n, "samples": count,
                                  "repeats": len(hits), "logit_contrast": math.log(n),
                                  "marked_probability": n / (2 * n - 1), "exact_scalar_output": exact,
                                  "mean_absolute_error": float(error.mean()),
                                  "success_at_0.1": float(np.mean(error <= 0.1)),
                                  "empirical_miss_probability": float(np.mean(hits == 0)),
                                  "analytic_miss_probability": (1 - 1 / n)**count})
    for d in config["resource_dimensions"]:
        for n in config["sequence_lengths"]:
            for cap in config["resource_caps"]:
                for regime, kappa in [("bounded", 1), ("logarithmic", math.log(n)),
                                      ("native_frobenius_bound", n * cap**2 / math.sqrt(d))]:
                    # Unit input rows and Frobenius access: |physical logits| <= kappa/N.
                    # Use this certified Z lower bound, not the weaker full-interval N*exp(-kappa).
                    log_z_lower = math.log(n) - kappa / n
                    log_delta = math.log(config["resource_row_state_error"] / 8) + (log_z_lower - math.log(n)) / 2
                    delta = math.exp(log_delta)
                    degree = taylor_degree(kappa, delta)
                    log_c_upper = float(np.logaddexp(0.5 * math.log(n), kappa / 2))
                    log_row_proxy = math.log(max(1, degree)) + log_c_upper - log_z_lower / 2
                    log_prefactor = 0.5 * math.log(n) + 4 * math.log(cap) - math.log(4)
                    resources.append({"N": n, "dimension": d, "alpha_w": cap, "alpha_m": cap,
                                      "regime": regime, "kappa": kappa, "degree_sufficient": degree,
                                      "polynomial_delta": delta, "log_Z_lower": log_z_lower,
                                      "log10_C_upper": log_c_upper / math.log(10),
                                      "log10_row_query_proxy": log_row_proxy / math.log(10),
                                      "weight_prefactor": cap**4, "conditioning_product_assumed": 4,
                                      "log10_layer_readout_proxy": (log_prefactor + log_row_proxy + math.log(d / config["resource_readout_error"])) / math.log(10)})
    save_csv(output / "attention.csv", measurements)
    save_csv(output / "sampler_trials.csv", errors)
    save_csv(output / "selective_sampler.csv", selective)
    save_csv(output / "resource_proxies.csv", resources)
    save_json(output / "manifest.json", manifest(config, [Path(__file__), ROOT / "experiments/lib/experiment.py"],
              seconds=time.perf_counter() - started, dtype="float64",
              limitations=["Synthetic inputs; no trained-model quality inference.",
                           "Selective experiment verifies only the scalar distribution, not a quantum implementation or lower bound.",
                           "Resource proxies omit unspecified leading constants and logarithmic overheads; they are not exact oracle totals."]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/configs/diagnostics/synthetic.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
