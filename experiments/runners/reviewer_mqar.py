"""Prepare, train, and evaluate the pinned MQAR benchmark without test selection."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
from torch.nn import functional as F

from experiments.lib.mqar import (
    MQARTransformer, VENDOR, generate_data, state_hash, validate_data, weight_records,
)
from experiments.lib.reviewer import manifest, save_csv, save_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return [Path(__file__), ROOT / "experiments/lib/mqar.py",
            ROOT / "experiments/lib/reviewer.py", VENDOR / "multiquery_ar.py",
            VENDOR / "UPSTREAM.json", ROOT / "experiments/configs/reviewer/MQAR_PROTOCOL.txt",
            ROOT / "src/quantum_friendly_transformer/norm_transformer/frobenius_cap.py",
            ROOT / "src/quantum_friendly_transformer/norm_transformer/normalization.py"]


def prepare(config, output, smoke=False):
    output.mkdir(parents=True, exist_ok=False)
    datasets = [
        ("train", config["train_length"], config["train_pairs"], config["train_examples"], config["train_data_seed"]),
        ("validation", config["train_length"], config["train_pairs"], config["validation_examples"], config["validation_data_seed"]),
    ]
    datasets += [(f"test_N{c['N']}_K{c['pairs']}", c["N"], c["pairs"], config["test_examples"],
                  config["test_data_seed"] + 100 * c["N"] + c["pairs"]) for c in config["test_conditions"]]
    records, row_sets = [], {}
    for name, n, pairs, count, seed in datasets:
        data = generate_data(config, n=n, pairs=pairs, examples=count, seed=seed)
        validate_data(data, pairs, config["vocab_size"])
        hashes = {hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in data.inputs}
        if len(hashes) != count:
            raise AssertionError("Duplicate complete sequence within a split")
        for previous, previous_hashes in row_sets.items():
            if hashes & previous_hashes:
                raise AssertionError(f"Exact sequence overlap: {previous} and {name}")
        row_sets[name] = hashes
        path = output / f"{name}.pt"
        torch.save({"inputs": data.inputs, "labels": data.labels, "N": n,
                    "pairs": pairs, "seed": seed, "slices": data.slices}, path)
        records.append({"name": name, "file": path.name, "N": n, "pairs": pairs,
                        "sequences": count, "scored_queries": count * pairs,
                        "data_seed": seed, "sha256": sha256(path)})
        print(f"DATA {name}: sequences={count} scored_queries={count*pairs}", flush=True)
    save_json(output / "config.json", config)
    save_json(output / "manifest.json", manifest(config, sources(), smoke=smoke,
        datasets=records, exact_sequence_overlap=False, validation="Every official query checked against prefix memory"))


def load_data(assets, name):
    data_manifest = json.loads((assets / "manifest.json").read_text())
    record = next(r for r in data_manifest["datasets"] if r["name"] == name)
    path = assets / record["file"]
    if sha256(path) != record["sha256"]:
        raise AssertionError(f"Dataset checksum mismatch: {name}")
    return torch.load(path, weights_only=True, map_location="cpu")


@torch.inference_mode()
def score(model, data, device, batch_size, *, predictions=None, ablation=False):
    model.eval()
    correct, total, nll_sum, exact_sequences, uniform_correct = 0, 0, 0., 0, 0
    diagnostics = [{"sigma1_min": math.inf, "sigma2_min": math.inf,
                    "input_row_norm_max": 0., "output_row_norm_max": 0.} for _ in model.blocks]
    stream = predictions.open("w", newline="") if predictions is not None else None
    writer = csv.DictWriter(stream, fieldnames=["sequence", "position", "label", "prediction", "uniform_prediction", "nll"]) if stream else None
    if writer:
        writer.writeheader()
    start_time = time.perf_counter()
    try:
        for start in range(0, len(data["inputs"]), batch_size):
            inputs = data["inputs"][start:start+batch_size].to(device)
            labels = data["labels"][start:start+batch_size].to(device)
            positions = (labels != -100).nonzero(as_tuple=True)
            target = labels[positions]
            logits, stats = model(inputs, positions, diagnostics=True)
            for summary, batch in zip(diagnostics, stats):
                for key in summary:
                    summary[key] = min(summary[key], batch[key]) if key.endswith("min") else max(summary[key], batch[key])
            predicted = logits.argmax(-1)
            nll = F.cross_entropy(logits, target, reduction="none")
            uniform_predicted = model(inputs, positions, force_uniform=True).argmax(-1) if ablation else predicted
            hits = predicted == target
            correct += int(hits.sum())
            uniform_correct += int((uniform_predicted == target).sum())
            total += len(target)
            nll_sum += float(nll.sum())
            exact_sequences += int(hits.reshape(len(inputs), data["pairs"]).all(-1).sum())
            if writer:
                arrays = [positions[0].cpu().tolist(), positions[1].cpu().tolist(), target.cpu().tolist(),
                          predicted.cpu().tolist(), uniform_predicted.cpu().tolist(), nll.cpu().tolist()]
                for seq, pos, label, prediction, uniform, loss in zip(*arrays):
                    writer.writerow({"sequence": start+seq, "position": pos, "label": label,
                                     "prediction": prediction, "uniform_prediction": uniform, "nll": loss})
    finally:
        if stream:
            stream.close()
    return {"N": data["N"], "pairs": data["pairs"], "queries": total,
            "sequences": len(data["inputs"]), "correct": correct, "accuracy": correct / total,
            "mean_nll": nll_sum / total, "all_queries_correct_sequences": exact_sequences,
            "sequence_accuracy": exact_sequences / len(data["inputs"]),
            "uniform_ablation_accuracy": uniform_correct / total if ablation else None,
            "layer_diagnostics": diagnostics, "evaluation_seconds": time.perf_counter()-start_time}


def train(config, assets, output, seed, variant, device, smoke=False):
    if config != json.loads((assets / "config.json").read_text()):
        raise ValueError("Data and run configurations differ")
    output.mkdir(parents=True, exist_ok=False)
    for source in sources():
        target = output / "source_snapshot" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    save_json(output / "config.json", config)
    provenance = manifest(config, sources(), seed=seed, variant=variant, smoke=smoke,
        device=str(device), dtype="float32", tf32=False,
        selection="Fixed final epoch; validation diagnostic only; all seeds retained",
        supervision="Official answer-token labels only; no routing/index auxiliary loss",
        data_manifest_sha256=sha256(assets / "manifest.json"))
    save_json(output / "manifest.json", provenance)
    training = load_data(assets, "train")
    validation = load_data(assets, "validation")
    torch.manual_seed(seed)
    model = MQARTransformer(config, variant).to(device)
    initial_sha = state_hash(model.state_dict())
    save_csv(output / "weights_initial.csv", weight_records(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 eps=config["adam_epsilon"], weight_decay=config["weight_decay"])
    batches_per_epoch = math.ceil(len(training["inputs"]) / config["batch_size"])
    total_steps = config["epochs"] * batches_per_epoch
    warmup = max(1, int(total_steps * config["warmup_fraction"]))
    order_rng = torch.Generator().manual_seed(seed + 100000)
    history, validation_history = [], []
    step = 0
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        order = torch.randperm(len(training["inputs"]), generator=order_rng)
        loss_sum, correct, total = 0., 0, 0
        model.train()
        for start in range(0, len(order), config["batch_size"]):
            step += 1
            if step <= warmup:
                factor = step / warmup
            else:
                progress = (step - warmup) / max(1, total_steps - warmup)
                factor = config["minimum_lr_fraction"] + (1-config["minimum_lr_fraction"]) * (1+math.cos(math.pi*progress))/2
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rate"] * factor
            indices = order[start:start+config["batch_size"]]
            inputs = training["inputs"][indices].to(device)
            labels = training["labels"][indices].to(device)
            positions = (labels != -100).nonzero(as_tuple=True)
            target = labels[positions]
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, positions)
            loss = F.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise AssertionError("Nonfinite training loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(target)
            correct += int((logits.detach().argmax(-1) == target).sum())
            total += len(target)
            if step == 1 or step % config["log_every"] == 0:
                print(f"TRAIN {output.name} epoch={epoch} step={step}/{total_steps} accuracy={correct/total:.4f} nll={loss_sum/total:.4f}", flush=True)
                save_json(output / "progress.json", {"epoch": epoch, "step": step, "total_steps": total_steps,
                    "seconds": time.perf_counter()-started, "training_accuracy": correct/total})
        history.append({"epoch": epoch, "step": step, "accuracy": correct/total,
                        "mean_nll": loss_sum/total, "lr": optimizer.param_groups[0]["lr"],
                        "last_gradient_norm": float(gradient), "seconds": time.perf_counter()-started})
        valid = score(model, validation, device, config["eval_batch_size"])
        validation_history.append({"epoch": epoch, **valid})
        save_csv(output / "training.csv", history)
        save_json(output / "validation.json", validation_history)
        save_csv(output / "weights_latest.csv", weight_records(model))
        print(f"VALID {output.name} epoch={epoch} accuracy={valid['accuracy']:.4f} nll={valid['mean_nll']:.4f}", flush=True)
    checkpoint = output / "final.pth"
    torch.save({"config": config, "seed": seed, "variant": variant,
                "state_dict": {k: v.detach().cpu() for k,v in model.state_dict().items()}}, checkpoint)
    state_sha = state_hash(model.state_dict())
    del optimizer, model
    restored = MQARTransformer(config, variant).to(device)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device)["state_dict"])
    if state_hash(restored.state_dict()) != state_sha:
        raise AssertionError("Checkpoint restoration mismatch")
    evaluations = []
    for condition in config["test_conditions"]:
        name = f"test_N{condition['N']}_K{condition['pairs']}"
        data = load_data(assets, name)
        metrics = score(restored, data, device, config["eval_batch_size"],
                        predictions=output / f"{name}_predictions.csv", ablation=True)
        evaluations.append({"role": condition["role"], **metrics})
        print(f"TEST {output.name} N={condition['N']} K={condition['pairs']} accuracy={metrics['accuracy']:.4f} uniform_ablation={metrics['uniform_ablation_accuracy']:.4f}", flush=True)
    save_csv(output / "weights_final.csv", weight_records(restored))
    save_json(output / "result.json", {"seed": seed, "variant": variant, "smoke": smoke,
        "initial_state_sha256": initial_sha, "checkpoint_sha256": sha256(checkpoint),
        "final_state_sha256": state_sha, "training_steps": step,
        "training_seconds_including_validation": history[-1]["seconds"],
        "total_seconds": time.perf_counter()-started, "evaluation": evaluations})
    save_json(output / "progress.json", {"status": "complete", "step": step, "total_steps": total_steps})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train"])
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/configs/reviewer/mqar.json")
    parser.add_argument("--assets", type=Path, default=ROOT / "results/reviewer/mqar_data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variant", choices=["native", "uniform"], default="native")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config["threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.smoke:
        config = copy.deepcopy(config)
        config.update(vocab_size=128, dimension=16, ffn_dimension=32, train_length=32, train_pairs=4,
                      train_examples=16, validation_examples=8, test_examples=8, epochs=1,
                      batch_size=8, eval_batch_size=4, test_conditions=[{"N":32,"pairs":4,"role":"smoke"}])
    if args.action == "prepare":
        prepare(config, args.output, args.smoke)
    else:
        train(config, args.assets, args.output, args.seed, args.variant, torch.device(args.device), args.smoke)


if __name__ == "__main__":
    main()
