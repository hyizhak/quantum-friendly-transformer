"""Train fixed-budget paired long-context recall experiments and retain every seed."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
from torch.nn import functional as F

from experiments.lib.long_recall import LongRecallLayer, input_certificate, matrix_records, paired_batch
from experiments.lib.experiment import manifest, save_csv, save_json


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_model(config, variant, device):
    return LongRecallLayer(config["dimension"], config["ffn_dimension"], config["cap"], variant).to(device)


@torch.inference_mode()
def evaluate(model, config, destination, device):
    model = model.double().eval()
    batch_rows, checks = [], []
    with (destination / "predictions.csv").open("w", newline="") as stream:
        fields = ["N", "example", "pair", "label", "prediction", "uniform_prediction",
                  "shuffled_query_prediction", "margin", "nll", "target_probability"]
        fields += [f"noise_{eps:g}_prediction" for eps in config["attention_noise_l2"]]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for n in config["test_lengths"]:
            generator = torch.Generator(device=device).manual_seed(config["test_seed"] + n)
            noise_generator = torch.Generator(device=device).manual_seed(config["test_seed"] + n + 10000000)
            for start in range(0, config["test_pairs"], config["test_batch_pairs"]):
                pairs = min(config["test_batch_pairs"], config["test_pairs"] - start)
                rows, labels, selected = paired_batch(pairs, n, config["dimension"], config["key_dimension"], generator, device)
                row_bound, matrix_bound = input_certificate(rows)
                rows = rows.double()
                mean_gap = float((rows[0::2].mean(1) - rows[1::2].mean(1)).abs().max())
                if mean_gap != 0:
                    raise AssertionError("Mean-pooling counterfactual certificate failed")
                logits, diag = model(rows, diagnostics=True)
                uniform_logits = model(rows, force_uniform=True)
                if not torch.equal(uniform_logits[0::2], uniform_logits[1::2]):
                    raise AssertionError("Uniform ablation differs on identical pooled inputs")
                shuffled = rows.clone()
                shuffled[:, -1, :config["key_dimension"]] = rows[:, -1, :config["key_dimension"]].roll(2, 0)
                shuffled_logits = model(shuffled)
                probability = diag["probabilities"]
                batch = torch.arange(len(labels), device=device)
                target_probability = probability[batch, selected]
                nll = F.cross_entropy(logits, labels, reduction="none")
                margin = logits[batch, labels] - logits[batch, 1 - labels]
                gap = (diag["hidden"][0::2] - diag["hidden"][1::2]).norm(dim=-1)
                prediction = logits.argmax(-1)
                noises = {}
                direction = torch.randn(pairs, model.d, generator=noise_generator, device=device, dtype=torch.float64)
                direction = F.normalize(direction, dim=-1).repeat_interleave(2, 0)
                for eps in config["attention_noise_l2"]:
                    noisy_logits, _, _ = model.finish(diag["attended"] + eps * direction, rows[:, -1])
                    noises[eps] = noisy_logits.argmax(-1)
                entropy = -(probability * probability.clamp_min(1e-300).log()).sum(-1)
                max_logit = float(diag["scores"].abs().max())
                if model.variant == "strict_frobenius" and max_logit > 1 / n + 1e-12:
                    raise AssertionError("Strict theorem logit bound exceeded")
                if not all(torch.isfinite(value).all() for value in (logits, margin, gap, diag["sigma1"], diag["sigma2"])):
                    raise AssertionError("Nonfinite evaluation output")
                record = {
                    "N": n, "start_pair": start, "examples": len(labels),
                    "correct": int((prediction == labels).sum()),
                    "uniform_correct": int((uniform_logits.argmax(-1) == labels).sum()),
                    "shuffled_query_correct": int((shuffled_logits.argmax(-1) == labels).sum()),
                    "mean_nll": float(nll.mean()), "mean_margin": float(margin.mean()),
                    "minimum_margin": float(margin.min()),
                    "mean_pair_hidden_gap": float(gap.mean()), "minimum_pair_hidden_gap": float(gap.min()),
                    "mean_target_probability": float(target_probability.mean()),
                    "mean_max_probability": float(probability.max(-1).values.mean()),
                    "mean_normalized_entropy": float(entropy.mean() / math.log(n)),
                    "maximum_absolute_logit": max_logit,
                    "maximum_logit_contrast": float((diag["scores"].max(-1).values - diag["scores"].min(-1).values).max()),
                    "minimum_sigma1": float(diag["sigma1"].min()), "minimum_sigma2": float(diag["sigma2"].min()),
                    "maximum_input_row_squared_norm": row_bound, "maximum_input_matrix_squared_norm": matrix_bound,
                    "maximum_paired_mean_gap": mean_gap,
                    "denominator": model.denominator(n) if model.variant != "uniform" else 0.,
                    "kappa": n * model.cap**2 / model.denominator(n) if model.variant != "uniform" else 0.,
                }
                record.update({f"noise_{eps:g}_correct": int((values == labels).sum()) for eps, values in noises.items()})
                batch_rows.append(record)
                if start == 0:
                    # Independently materialize all K and V projections on a small
                    # set; also check a key permutation, without modifying the query.
                    direct, direct_hidden, _ = model.dense_forward(rows[:4])
                    dense_error = float((direct - logits[:4]).abs().max())
                    hidden_error = float((direct_hidden - diag["hidden"][:4]).abs().max())
                    permutation = torch.arange(n, device=device)
                    permutation[:-1] = permutation[:-1].flip(0)
                    permuted = model(rows[:4, permutation])
                    permutation_error = float((permuted - logits[:4]).abs().max())
                    if max(dense_error, hidden_error, permutation_error) > 1e-9:
                        raise AssertionError("Independent dense/permutation check failed")
                    checks.append({"N": n, "dense_logit_max_error": dense_error,
                                   "dense_hidden_max_error": hidden_error, "permutation_max_error": permutation_error})
                arrays = {
                    "label": labels.cpu().tolist(), "prediction": prediction.cpu().tolist(),
                    "uniform_prediction": uniform_logits.argmax(-1).cpu().tolist(),
                    "shuffled_query_prediction": shuffled_logits.argmax(-1).cpu().tolist(),
                    "margin": margin.cpu().tolist(), "nll": nll.cpu().tolist(),
                    "target_probability": target_probability.cpu().tolist(),
                    **{f"noise_{eps:g}_prediction": values.cpu().tolist() for eps, values in noises.items()},
                }
                for i in range(len(labels)):
                    writer.writerow({"N": n, "example": 2 * start + i, "pair": start + i // 2,
                                     **{key: value[i] for key, value in arrays.items()}})
            current = [r for r in batch_rows if r["N"] == n]
            accuracy = sum(r["correct"] for r in current) / sum(r["examples"] for r in current)
            print(f"TEST {destination.name} N={n} accuracy={accuracy:.4f}", flush=True)
            save_csv(destination / "test.csv", batch_rows)
    save_json(destination / "independent_forward_checks.json", checks)
    save_csv(destination / "weights_evaluation_float64.csv", matrix_records(model))
    return batch_rows


def run_one(config, output, seed, variant, device, smoke):
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / "experiments/lib/long_recall.py", ROOT / "experiments/lib/experiment.py",
               ROOT / "src/quantum_friendly_transformer/norm_transformer/frobenius_cap.py",
               ROOT / "src/quantum_friendly_transformer/norm_transformer/normalization.py"]
    provenance = manifest(config, sources, seed=seed, variant=variant, smoke=smoke,
        training_dtype="float32", evaluation_dtype="float64", device=str(device),
        selection="Fixed final step only; no test checkpoint/protocol selection",
        task="Fresh continuous-key binary recall, counterfactual pairs with exactly equal row means",
        scope="General-alpha1 is the general-scaling regime; only strict_frobenius enforces alpha0=N*24^2")
    save_json(output / "running_manifest.json", provenance)
    save_json(output / "config.json", config)
    torch.manual_seed(seed)
    model = make_model(config, variant, device)
    initial_hash = state_hash(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 eps=config["adam_epsilon"], weight_decay=config["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(seed + 100000)
    history, weight_history = [], []
    loss_sum, correct_sum, seen = 0., 0, 0
    maximum_row, maximum_matrix_ratio = 0., 0.
    sigma1_min, sigma2_min = math.inf, math.inf
    started = time.perf_counter()
    model.train()
    for step in range(1, config["steps"] + 1):
        n = next(item["N"] for item in config["curriculum"] if step <= item["through_step"])
        rows, labels, _ = paired_batch(config["batch_pairs"], n, config["dimension"], config["key_dimension"], generator, device)
        row_bound, matrix_bound = input_certificate(rows)
        maximum_row = max(maximum_row, row_bound)
        maximum_matrix_ratio = max(maximum_matrix_ratio, matrix_bound / n)
        optimizer.zero_grad(set_to_none=True)
        logits, diag = model(rows, diagnostics=True)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite training loss")
        sigma1_min = min(sigma1_min, float(diag["sigma1"].detach().min()))
        sigma2_min = min(sigma2_min, float(diag["sigma2"].detach().min()))
        if variant == "strict_frobenius" and float(diag["scores"].detach().abs().max()) > 1 / n + 1e-7:
            raise AssertionError("Training logit bound exceeded")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        loss_sum += float(loss.detach()) * len(labels)
        correct_sum += int((logits.detach().argmax(-1) == labels).sum())
        seen += len(labels)
        if step == 1 or step % config["log_every"] == 0 or step == config["steps"]:
            history.append({"step": step, "N": n, "examples": seen,
                            "mean_loss": loss_sum / seen, "accuracy": correct_sum / seen,
                            "gradient_norm": float(gradient), "minimum_sigma1_so_far": sigma1_min,
                            "minimum_sigma2_so_far": sigma2_min,
                            "maximum_input_row_squared_norm_so_far": maximum_row,
                            "maximum_input_matrix_squared_to_N_so_far": maximum_matrix_ratio})
            weight_history.extend({"step": step, **r} for r in matrix_records(model))
            print(f"TRAIN {output.name} step={step} N={n} accuracy={correct_sum / seen:.4f} loss={loss_sum / seen:.5f}", flush=True)
            save_csv(output / "training.csv", history)
            save_csv(output / "weights_training.csv", weight_history)
            save_json(output / "progress.json", {"step": step, "N": n, "seconds": time.perf_counter() - started})
            loss_sum, correct_sum, seen = 0., 0, 0
    checkpoint = {"config": config, "seed": seed, "variant": variant,
                  "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    torch.save(checkpoint, output / "final.pth")
    checksum = hashlib.sha256((output / "final.pth").read_bytes()).hexdigest()
    restored = make_model(config, variant, device)
    restored.load_state_dict(torch.load(output / "final.pth", map_location=device, weights_only=True)["state_dict"])
    if state_hash(restored.state_dict()) != state_hash(model.state_dict()):
        raise AssertionError("Checkpoint restoration mismatch")
    del optimizer, model
    metrics = evaluate(restored, config, output, device)
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
        "initial_state_sha256": initial_hash, "checkpoint_sha256": checksum,
        "training_input_row_squared_max": maximum_row, "training_input_matrix_squared_to_N_max": maximum_matrix_ratio,
        "training_sigma1_min": sigma1_min, "training_sigma2_min": sigma2_min,
        "seconds": time.perf_counter() - started, "evaluation": summary})
    save_json(output / "manifest.json", provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.smoke:
        config = copy.deepcopy(config)
        config.update(seeds=[11], steps=3, log_every=1, test_pairs=8, test_batch_pairs=4,
                      test_lengths=[64, 128], curriculum=[{"through_step": 3, "N": 64}])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "config.json", config)
    plan = [{"seed": seed, "variant": variant, "status": "planned"} for seed in config["seeds"] for variant in config["variants"]]
    save_json(args.output / "plan.json", plan)
    for item in plan:
        item["status"] = "running"
        save_json(args.output / "plan.json", plan)
        run_one(config, args.output / f"{item['variant']}_seed{item['seed']}", item["seed"], item["variant"], device, args.smoke)
        item["status"] = "completed"
        save_json(args.output / "plan.json", plan)


if __name__ == "__main__":
    main()
