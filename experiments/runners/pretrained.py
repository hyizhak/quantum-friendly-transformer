"""Run paired context-length and bounded interface-noise evaluations on GPT-Neo."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantum_friendly_transformer import apply_attention_scaling, apply_learnable_frobenius_caps, apply_normalization_affine_constraints
from quantum_friendly_transformer.norm_transformer.frobenius_cap import target_weight_modules
from experiments.lib.experiment import manifest, save_csv, save_json
from experiments.lib.pretrained import GPTNeoDiagnostics, InterfaceNoise


def weight_stats(model, variant):
    rows = []
    for name, module in target_weight_modules(model):
        effective = float(module.weight.detach().float().norm())
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is not None and "weight" in parametrizations:
            raw = float(parametrizations.weight.original.detach().float().norm())
            gamma = float(parametrizations.weight[0].gamma.detach())
        else:
            raw, gamma = effective, None
        rows.append({"variant": variant, "matrix": name, "shape": str(list(module.weight.shape)),
                     "raw_frobenius_norm": raw, "effective_frobenius_norm": effective,
                     "gamma": gamma if gamma is not None else "", "cap_active": int(raw > gamma) if gamma is not None else 0})
    return rows


def load_model(assets, variant, n, cap):
    model = AutoModelForCausalLM.from_pretrained(assets / "model", local_files_only=True,
                                                attn_implementation="eager").eval()
    if variant != "native":
        count = apply_learnable_frobenius_caps(model, max_gamma=cap)
        if count != len(model.transformer.h) * 6:
            raise AssertionError(f"Incomplete cap coverage: {count}")
    if variant.startswith("constrained_"):
        # Constrain transformer block norms only. Keep the pretrained output
        # normalization and LM head, avoiding a trivial readout-temperature change.
        count = sum(apply_normalization_affine_constraints(block) for block in model.transformer.h)
        if count != len(model.transformer.h) * 2:
            raise AssertionError(f"Incomplete block normalization coverage: {count}")
    if variant == "constrained_theorem":
        records = apply_attention_scaling(model, "alpha_s_alpha_w", alpha_s=math.sqrt(n), alpha_w=cap)
        if len(records) != len(model.transformer.h):
            raise AssertionError("Incomplete attention scaling coverage")
    if variant == "constrained_uniform":
        for block in model.transformer.h:
            block.attn.attention.q_proj.register_forward_hook(lambda module, args, output: torch.zeros_like(output))
    model.eval()
    return model


def score(model, tokens, score_tokens):
    hidden = model.transformer(input_ids=tokens, use_cache=False).last_hidden_state
    logits = model.lm_head(hidden[:, -score_tokens - 1:-1])
    labels = tokens[:, -score_tokens:]
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none")
    accuracy = float((logits.argmax(-1) == labels).float().mean())
    return float(losses.double().mean()), accuracy


def run(config, assets, output, suite):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["threads"])
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    asset_manifest = json.loads((assets / "assets.json").read_text())
    texts = []
    for file in asset_manifest["data_files"]:
        texts.extend(pq.read_table(assets / "data" / file, columns=["text"])["text"].to_pylist())
    tokenizer = AutoTokenizer.from_pretrained(assets / "model", local_files_only=True)
    ids = tokenizer("\n\n".join(texts), add_special_tokens=False, verbose=False)["input_ids"]
    max_n = max(config["sequence_lengths"])
    assert len(ids) >= config["blocks"] * max_n
    windows = [torch.tensor(ids[i * max_n:(i + 1) * max_n], dtype=torch.long).unsqueeze(0)
               for i in range(config["blocks"])]
    if config["score_tokens"] >= min(config["sequence_lengths"]):
        raise ValueError("Scored target count must be smaller than every context")
    losses, weights, attention, normalization, residual, inputs, noise_rows = [], [], [], [], [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        if suite in {"all", "scaling"}:
            for variant in config["variants"]:
                for n in config["sequence_lengths"]:
                    model = load_model(assets, variant, n, config["cap"])
                    if n > model.config.max_position_embeddings:
                        raise ValueError("Requested context exceeds the model's trained position limit")
                    if n == min(config["sequence_lengths"]):
                        weights.extend(weight_stats(model, variant))
                    diag = GPTNeoDiagnostics(model, variant, n, math.sqrt(n) if variant.startswith("constrained_") else None)
                    for index, window in enumerate(windows):
                        diag.context["block"] = index
                        loss, accuracy = score(model, window[:, -n:], config["score_tokens"])
                        losses.append({"variant": variant, "N": n, "block": index,
                                       "scored_tokens": config["score_tokens"], "mean_nll": loss,
                                       "perplexity": math.exp(loss), "next_token_accuracy": accuracy})
                    attention.extend(diag.attention)
                    normalization.extend(diag.normalization)
                    residual.extend(diag.residual)
                    inputs.extend(diag.inputs)
                    diag.close()
                    del model
                    save_csv(output / "losses.csv", losses)
                    save_csv(output / "attention.csv", attention)
                    save_csv(output / "normalization.csv", normalization)
                    save_csv(output / "residual.csv", residual)
                    save_csv(output / "attention_inputs.csv", inputs)
                    save_csv(output / "weights.csv", weights)
                    mean_loss = np.mean([r["mean_nll"] for r in losses if r["variant"] == variant and r["N"] == n])
                    print(f"{variant} N={n}: paired-window perplexity={math.exp(mean_loss):.4f}", flush=True)
        if suite in {"all", "noise"}:
            n = config["noise_sequence_length"]
            model = load_model(assets, "native", n, config["cap"])
            baseline = {}
            for index, window in enumerate(windows[:config["noise_blocks"]]):
                baseline[index] = score(model, window[:, -n:], config["score_tokens"])[0]
                noise_rows.append({"noise_type": "none", "epsilon_rel": 0, "perturbed_layers": 0,
                                   "seed": 0, "N": n, "block": index, "mean_nll": baseline[index],
                                   "delta_nll": 0, "max_bound_ratio": 0})
            for kind in config["noise_types"]:
                for epsilon in config["noise_relative_errors"]:
                    for depth in config["noise_depths"]:
                        for seed in config["noise_seeds"] if kind == "uniform" else [0]:
                            noise = InterfaceNoise(model, depth, epsilon, kind, seed)
                            for index, window in enumerate(windows[:config["noise_blocks"]]):
                                loss, _ = score(model, window[:, -n:], config["score_tokens"])
                                noise_rows.append({"noise_type": kind, "epsilon_rel": epsilon,
                                                   "perturbed_layers": depth, "seed": seed, "N": n,
                                                   "block": index, "mean_nll": loss,
                                                   "delta_nll": loss - baseline[index],
                                                   "max_bound_ratio": noise.max_bound_ratio})
                            noise.close()
                        save_csv(output / "noise.csv", noise_rows)
                        mean_delta = np.mean([r["delta_nll"] for r in noise_rows if r["noise_type"] == kind and r["epsilon_rel"] == epsilon and r["perturbed_layers"] == depth])
                        print(f"noise={kind} epsilon_rel={epsilon} layers={depth}: delta_nll={mean_delta:.5f}", flush=True)
    save_json(output / "manifest.json", manifest(config, [Path(__file__), ROOT / "experiments/lib/experiment.py", ROOT / "experiments/lib/pretrained.py",
              ROOT / "src/quantum_friendly_transformer/norm_transformer/attention_scaling.py",
              ROOT / "src/quantum_friendly_transformer/norm_transformer/normalization.py",
              ROOT / "src/quantum_friendly_transformer/norm_transformer/frobenius_cap.py"],
              assets=asset_manifest, suite=suite, device="cpu", dtype="float32", seconds=time.perf_counter() - started,
              token_stream_sha256=hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest(),
              token_stream_length=len(ids), windows="First disjoint maximum-length blocks of the validation token stream; shorter contexts are suffixes; score the same final target tokens.",
              limitations=["Frozen pretrained model interventions; no fine-tuning or training-seed comparison.",
                           "GPT-Neo is a pre-LayerNorm multihead decoder with local/global layers, not the theorem's complete layer architecture.",
                           "Only transformer block LayerNorm affine parameters are constrained; final output normalization, embeddings, biases, and LM head are retained.",
                           "Reported perplexity covers fixed paired target windows, not the full WikiText-2 validation corpus.",
                           "Noise is bounded in each residual row's unit-norm coordinates and is a sensitivity surrogate, not simulated quantum tomography.",
                           "Theorem scaling uses a fixed full-context Frobenius bound sqrt(N) and cap 24; no cache reuse across changed contexts."]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/configs/diagnostics/pretrained.json")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["all", "scaling", "noise"], default="all")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.assets, args.output, args.suite)


if __name__ == "__main__":
    main()
