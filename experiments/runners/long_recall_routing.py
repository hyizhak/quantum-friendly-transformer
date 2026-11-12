"""Exploratory matched-location auxiliary supervision for the frozen recall task.

The matching index is supplied only to the training loss. Inference receives
exactly the same rows as the classification-only experiment.
"""

import hashlib
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
from torch.nn import functional as F

from experiments.runners import long_recall as core
from experiments.lib.long_recall import input_certificate, matrix_records, paired_batch
from experiments.lib.experiment import manifest, save_csv, save_json


def run_one(config, output, seed, variant, device, smoke):
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(core.__file__), ROOT / "experiments/lib/long_recall.py",
               ROOT / "experiments/lib/experiment.py",
               ROOT / "src/quantum_friendly_transformer/norm_transformer/frobenius_cap.py",
               ROOT / "src/quantum_friendly_transformer/norm_transformer/normalization.py"]
    provenance = manifest(config, sources, seed=seed, variant=variant, smoke=smoke,
        training_dtype="float32", evaluation_dtype="float64", device=str(device),
        selection="Fixed final step; protocol follow-up motivated by early classification-only results",
        auxiliary_supervision="Ground-truth matching-token position from synthetic TRAINING batches only",
        inference="Rows only; target index is never an input to model.forward",
        scope="General-alpha1 uses the general-scaling regime, not the strict Frobenius denominator")
    save_json(output / "running_manifest.json", provenance)
    save_json(output / "config.json", config)
    torch.manual_seed(seed)
    model = core.make_model(config, variant, device)
    initial_hash = core.state_hash(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 eps=config["adam_epsilon"], weight_decay=config["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(seed + 100000)
    history, weight_history = [], []
    loss_sum, route_sum, correct_sum, seen = 0., 0., 0, 0
    maximum_row, maximum_matrix_ratio = 0., 0.
    sigma1_min, sigma2_min = math.inf, math.inf
    started = time.perf_counter()
    model.train()
    for step in range(1, config["steps"] + 1):
        n = next(item["N"] for item in config["curriculum"] if step <= item["through_step"])
        rows, labels, selected = paired_batch(config["batch_pairs"], n, config["dimension"], config["key_dimension"], generator, device)
        row_bound, matrix_bound = input_certificate(rows)
        maximum_row = max(maximum_row, row_bound)
        maximum_matrix_ratio = max(maximum_matrix_ratio, matrix_bound / n)
        optimizer.zero_grad(set_to_none=True)
        logits, diag = model(rows, diagnostics=True)
        classification_loss = F.cross_entropy(logits, labels)
        # Uniform scores are constant: this term is log(N) with no gradient.
        route_loss = F.cross_entropy(diag["scores"], selected)
        loss = classification_loss + config["routing_auxiliary_weight"] * route_loss
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite training loss")
        sigma1_min = min(sigma1_min, float(diag["sigma1"].detach().min()))
        sigma2_min = min(sigma2_min, float(diag["sigma2"].detach().min()))
        if variant == "strict_frobenius" and float(diag["scores"].detach().abs().max()) > 1 / n + 1e-7:
            raise AssertionError("Training logit bound exceeded")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        loss_sum += float(classification_loss.detach()) * len(labels)
        route_sum += float(route_loss.detach()) * len(labels)
        correct_sum += int((logits.detach().argmax(-1) == labels).sum())
        seen += len(labels)
        if step == 1 or step % config["log_every"] == 0 or step == config["steps"]:
            history.append({"step": step, "N": n, "examples": seen,
                            "mean_loss": loss_sum / seen, "mean_route_loss": route_sum / seen,
                            "accuracy": correct_sum / seen, "gradient_norm": float(gradient),
                            "minimum_sigma1_so_far": sigma1_min, "minimum_sigma2_so_far": sigma2_min,
                            "maximum_input_row_squared_norm_so_far": maximum_row,
                            "maximum_input_matrix_squared_to_N_so_far": maximum_matrix_ratio})
            weight_history.extend({"step": step, **r} for r in matrix_records(model))
            print(f"TRAIN {output.name} step={step} N={n} accuracy={correct_sum / seen:.4f} route={route_sum / seen:.4f}", flush=True)
            save_csv(output / "training.csv", history)
            save_csv(output / "weights_training.csv", weight_history)
            save_json(output / "progress.json", {"step": step, "N": n, "seconds": time.perf_counter() - started})
            loss_sum, route_sum, correct_sum, seen = 0., 0., 0, 0
    torch.save({"config": config, "seed": seed, "variant": variant,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, output / "final.pth")
    checksum = hashlib.sha256((output / "final.pth").read_bytes()).hexdigest()
    restored = core.make_model(config, variant, device)
    restored.load_state_dict(torch.load(output / "final.pth", map_location=device, weights_only=True)["state_dict"])
    if core.state_hash(restored.state_dict()) != core.state_hash(model.state_dict()):
        raise AssertionError("Checkpoint restoration mismatch")
    del optimizer, model
    metrics = core.evaluate(restored, config, output, device)
    summary = []
    for n in config["test_lengths"]:
        data = [r for r in metrics if r["N"] == n]
        count = sum(r["examples"] for r in data)
        summary.append({"N": n, "examples": count, "accuracy": sum(r["correct"] for r in data) / count,
                        "uniform_accuracy": sum(r["uniform_correct"] for r in data) / count,
                        "shuffled_query_accuracy": sum(r["shuffled_query_correct"] for r in data) / count,
                        **{f"noise_{eps:g}_accuracy": sum(r[f"noise_{eps:g}_correct"] for r in data) / count for eps in config["attention_noise_l2"]}})
    save_json(output / "result.json", {"seed": seed, "variant": variant, "smoke": smoke,
        "training_maximum_N": config["curriculum"][-1]["N"], "training_steps": config["steps"],
        "routing_auxiliary_weight": config["routing_auxiliary_weight"],
        "initial_state_sha256": initial_hash, "checkpoint_sha256": checksum,
        "training_input_row_squared_max": maximum_row, "training_input_matrix_squared_to_N_max": maximum_matrix_ratio,
        "training_sigma1_min": sigma1_min, "training_sigma2_min": sigma2_min,
        "seconds": time.perf_counter() - started, "evaluation": summary})
    save_json(output / "manifest.json", provenance)


if __name__ == "__main__":
    core.run_one = run_one
    core.main()
