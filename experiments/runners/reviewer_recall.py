"""Train paired multi-seed recall models with standard, theorem, and uniform attention."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch
from torch.nn import functional as F

from experiments.lib.reviewer import manifest, save_csv, save_json
from experiments.lib.reviewer_recall import RecallLayer, matrix_stats, recall_batch


def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    save_json(output / "config.json", config)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use an accessible GPU or set device=cpu in the config")
    training, test, weights = [], [], []
    started = time.perf_counter()
    for seed in config["seeds"]:
        for variant in config["variants"]:
            torch.manual_seed(seed)
            model = RecallLayer(config["dimension"], config["ffn_dimension"], config["classes"], config["cap"], variant).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            generator = torch.Generator(device="cpu").manual_seed(seed + 100000)
            running_loss, running_correct, running_count = 0.0, 0, 0
            model.train()
            for step in range(1, config["steps"] + 1):
                rows, labels, _ = recall_batch(config["batch_size"], config["train_sequence_length"],
                    config["dimension"], config["key_dimension"], config["classes"], generator)
                rows, labels = rows.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(rows)
                loss = F.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach()) * len(labels)
                running_correct += int((logits.detach().argmax(-1) == labels).sum())
                running_count += len(labels)
                if step == 1 or step % config["log_every"] == 0:
                    matrix_stats(model)
                    training.append({"seed": seed, "variant": variant, "step": step,
                                     "mean_training_loss": running_loss / running_count,
                                     "training_accuracy": running_correct / running_count})
                    print(f"recall seed={seed} {variant} step={step}: accuracy={running_correct / running_count:.4f}", flush=True)
                    running_loss, running_correct, running_count = 0.0, 0, 0
                    save_csv(output / "training.csv", training)
            model.eval()
            for row in matrix_stats(model):
                weights.append({"seed": seed, "variant": variant, **row})
            torch.save({"config": config, "seed": seed, "variant": variant,
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
                       output / "checkpoints" / f"{variant}_seed_{seed}.pth")
            with torch.inference_mode():
                for n in config["test_sequence_lengths"]:
                    test_generator = torch.Generator(device="cpu").manual_seed(config["test_seed"] + n)
                    for batch_index, start in enumerate(range(0, config["test_examples"], config["test_batch_size"])):
                        size = min(config["test_batch_size"], config["test_examples"] - start)
                        rows, labels, selected = recall_batch(size, n, config["dimension"], config["key_dimension"], config["classes"], test_generator)
                        if not torch.allclose(rows.norm(dim=-1), torch.ones(size, n), atol=2e-6):
                            raise AssertionError("Input rows are not unit norm")
                        logits, diag = model(rows.to(device), diagnostics=True)
                        probs = diag["probabilities"]
                        batch = torch.arange(size, device=device)
                        if variant == "theorem_frobenius" and float(diag["logits"].abs().max()) > 1 / n + 1e-6:
                            raise AssertionError("Theorem logit bound violated")
                        test.append({"seed": seed, "variant": variant, "N": n, "batch": batch_index,
                                     "examples": size, "correct": int((logits.argmax(-1).cpu() == labels).sum()),
                                     "mean_nll": float(F.cross_entropy(logits, labels.to(device))),
                                     "mean_matching_probability": float(probs[batch, selected.to(device)].mean()),
                                     "mean_max_probability": float(probs.max(-1).values.mean()),
                                     "mean_normalized_entropy": float((-(probs * probs.clamp_min(1e-30).log()).sum(-1) / math.log(n - 1)).mean()),
                                     "max_absolute_logit": float(diag["logits"].abs().max()),
                                     "minimum_sigma1": float(diag["sigma1"].min()),
                                     "minimum_sigma2": float(diag["sigma2"].min())})
                    selected_rows = [r for r in test if r["seed"] == seed and r["variant"] == variant and r["N"] == n]
                    accuracy = sum(r["correct"] for r in selected_rows) / sum(r["examples"] for r in selected_rows)
                    print(f"recall TEST seed={seed} {variant} N={n}: accuracy={accuracy:.4f}", flush=True)
                    save_csv(output / "test.csv", test)
                    save_csv(output / "weights.csv", weights)
    save_json(output / "manifest.json", manifest(config,
              [Path(__file__), ROOT / "experiments/lib/reviewer.py", ROOT / "experiments/lib/reviewer_recall.py",
               ROOT / "src/quantum_friendly_transformer/norm_transformer/frobenius_cap.py",
               ROOT / "src/quantum_friendly_transformer/norm_transformer/normalization.py"],
              seconds=time.perf_counter() - started, device=str(device),
              device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", dtype="float32",
              selection="Fixed final training step; test set never used for checkpoint selection.",
              data="Fresh CPU-generated random continuous keys and random discrete values; shared training batches across variants within each seed; shared held-out test batches across all seeds and variants.",
              limitations=["A controlled, small, single-layer task; not pretrained language modeling or an SST-2 replication.",
                           "Finite epsilon LayerNorm and a capped downstream classifier are explicit experimental choices.",
                           "Failure of a variant under this fixed training protocol is not an impossibility result for all architectures or optimizers.",
                           "Longer contexts are a length-generalization test; all models train at N=64."]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/configs/reviewer/recall.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)
