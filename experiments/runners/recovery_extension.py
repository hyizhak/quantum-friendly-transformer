"""Train a bounded SST-2 recovery run with independent checkpoint selection.

Historical alpha constants are held fixed. They are scalar design constants,
not certified input-matrix bounds; norm diagnostics explicitly test that gap.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from experiments.lib.experiment import load_config
from experiments.lib.recovery_extension import (
    AttentionInputMonitor, indices_hash, install_uniform_attention,
    partition_training, validate_scale_records,
)
from experiments.lib.experiment import manifest, save_csv, save_json
from quantum_friendly_transformer.experiments.runners.recovery.glue_sequence_classifier import make_decoder_sequence_classifier
from quantum_friendly_transformer.norm_transformer import (
    apply_attention_scaling, apply_learnable_frobenius_caps,
    collect_attention_scaling_stats, collect_learnable_cap_stats, project_attention_scales_,
)


def prepare_data(assets, model_name, config):
    tokenizer = AutoTokenizer.from_pretrained(assets / model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train = pq.read_table(assets / "data/sst2/train-00000-of-00001.parquet").to_pylist()
    evaluation = pq.read_table(assets / "data/sst2/validation-00000-of-00001.parquet").to_pylist()
    split = partition_training(train, evaluation, config["selection_fraction"], config["selection_split_seed"])
    datasets = {}
    for name, rows in [("train", [train[i] for i in split["train"]]),
                       ("selection", [train[i] for i in split["selection"]]),
                       ("evaluation", evaluation)]:
        encoded = tokenizer([r["sentence"] for r in rows], padding="max_length", truncation=True,
                            max_length=config["sequence_length"], return_tensors="pt")
        datasets[name] = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
                          "labels": torch.tensor([r["label"] for r in rows]),
                          "example_id": torch.tensor([r["idx"] for r in rows])}
    metadata = {"indices": split, "sha256": {k: indices_hash(v) for k, v in split.items()},
                "counts": {name: len(rows["labels"]) for name, rows in datasets.items()},
                "visible_length_mean": {name: float(rows["attention_mask"].sum(-1).float().mean()) for name, rows in datasets.items()}}
    return tokenizer, datasets, metadata


def build_model(assets, model_name, config, seed, mode, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    backbone = AutoModel.from_pretrained(assets / model_name, local_files_only=True,
                                         torch_dtype=torch.float32, attn_implementation="eager")
    backbone.config.use_cache = False
    backbone.config.problem_type = "single_label_classification"
    model = make_decoder_sequence_classifier(torch, nn, SequenceClassifierOutput, backbone, 2)
    applied = apply_learnable_frobenius_caps(model, config["cap"], norm_dtype="float32")
    alpha = config["models"][model_name]["alpha_s"]
    if mode == "bounded":
        adapters = apply_attention_scaling(model, "alpha_s_alpha_w", alpha_s=alpha, alpha_w=config["cap"],
                                          trainable=True, enforce_alpha_denominator_bound=True)
    else:
        adapters = [{"uniform_attention_layers": install_uniform_attention(model)}]
    return model.to(device), applied, adapters


def batch_at(data, indices):
    return {k: v[indices] for k, v in data.items() if k != "example_id"}


@torch.no_grad()
def evaluate(model, data, batch_size, alpha_s, device, monitor=False):
    model.eval()
    predictions, losses = [], []
    context = AttentionInputMonitor(model, alpha_s) if monitor else None
    if context:
        context.__enter__()
    try:
        for start in range(0, len(data["labels"]), batch_size):
            batch = batch_at(data, slice(start, start + batch_size))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**batch).logits.float()
            losses.append(torch.nn.functional.cross_entropy(logits, batch["labels"], reduction="none").cpu())
            predictions.append(logits.argmax(-1).cpu())
    finally:
        if context:
            context.__exit__()
    predicted = torch.cat(predictions)
    per_example_loss = torch.cat(losses)
    labels = data["labels"].cpu()
    scores = {"accuracy": float((predicted == labels).float().mean()),
              "correct": int((predicted == labels).sum()), "examples": len(labels),
              "mean_nll": float(per_example_loss.mean())}
    return scores, predicted, per_example_loss, context.rows() if context else []


def run(args):
    config = load_config(args.config, getattr(args, "checkpoint_root", None))
    if args.steps is not None:
        config["models"][args.model]["steps"] = args.steps
    if args.eval_every is not None:
        config["eval_every"] = args.eval_every
    model_config = config["models"][args.model]
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["threads"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this run configuration")
    save_json(args.output / "config.json", config)
    tokenizer, data, split = prepare_data(args.assets, args.model, config)
    save_json(args.output / "split.json", split)
    if args.smoke:
        data = {name: {key: values[:128] for key, values in rows.items()} for name, rows in data.items()}
    data = {name: {key: value.to(device) for key, value in rows.items()} for name, rows in data.items()}
    model, applied_caps, adapters = build_model(args.assets, args.model, config, args.seed, args.mode, device)
    model.config.pad_token_id = tokenizer.pad_token_id
    scale_params, other_params = [], []
    for name, param in model.named_parameters():
        (scale_params if "_qft_learnable_attention_scale." in name else other_params).append(param)
    groups = [{"params": other_params, "lr": model_config["learning_rate"], "weight_decay": config["weight_decay"], "eps": 1e-8}]
    if scale_params:
        groups.append({"params": scale_params, "lr": config["scale_learning_rate"], "weight_decay": 0.0, "eps": 1e-12})
    optimizer = torch.optim.AdamW(groups, fused=device.type == "cuda")
    train_model = torch.compile(model, mode="reduce-overhead", dynamic=False) if args.compile else model
    alpha_s = model_config["alpha_s"]
    evaluations, training, norm_rows, scale_rows, weight_rows = [], [], [], [], []
    best_key = (-math.inf, -math.inf)
    best_step = None
    selected_path = args.output / "selected.pth"
    start_time = time.monotonic()

    def record_selection(step):
        nonlocal best_key, best_step
        scores, _, _, norms = evaluate(model, data["selection"], config["eval_batch_size"], alpha_s, device, monitor=True)
        scales = collect_attention_scaling_stats(model)
        if args.mode == "bounded":
            validate_scale_records(scales, alpha_s, config["cap"])
        caps = collect_learnable_cap_stats(model)
        evaluations.append({"step": step, "seconds": time.monotonic() - start_time, **scores})
        norm_rows.extend({"step": step, **r} for r in norms)
        scale_rows.extend({"step": step, **r} for r in scales)
        weight_rows.extend({"step": step, **r} for r in caps)
        key = (scores["accuracy"], -scores["mean_nll"])
        if key > best_key:
            best_key, best_step = key, step
            torch.save({"model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "step": step, "selection_metrics": scores, "model": args.model,
                        "mode": args.mode, "seed": args.seed, "config": config}, selected_path)
        for name, rows in [("selection", evaluations), ("attention_inputs", norm_rows), ("scales", scale_rows), ("weights", weight_rows)]:
            save_csv(args.output / f"{name}.csv", rows)
        print(f"{args.model} {args.mode} seed={args.seed} step={step} selection={scores['accuracy']:.5f} best_step={best_step} elapsed={time.monotonic()-start_time:.1f}s", flush=True)
        model.train()

    record_selection(0)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 8917)
    size = len(data["train"]["labels"])
    order, cursor = torch.randperm(size, generator=generator).to(device), 0
    steps, warmup = model_config["steps"], max(1, int(model_config["steps"] * config["warmup_fraction"]))
    block_loss, block_correct, block_examples = [], [], 0
    loop_start = time.monotonic()
    for step in range(1, steps + 1):
        if cursor + config["batch_size"] > size:
            order, cursor = torch.randperm(size, generator=generator).to(device), 0
        indices = order[cursor:cursor + config["batch_size"]]
        cursor += config["batch_size"]
        batch = batch_at(data["train"], indices)
        multiplier = step / warmup if step <= warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
        optimizer.param_groups[0]["lr"] = model_config["learning_rate"] * multiplier
        if scale_params:
            optimizer.param_groups[1]["lr"] = config["scale_learning_rate"] * multiplier
        optimizer.zero_grad(set_to_none=True)
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = train_model(**batch)
            loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(other_params, 1.0)
        if scale_params:
            torch.nn.utils.clip_grad_norm_(scale_params, 1.0)
        optimizer.step()
        project_attention_scales_(model, optimizer)
        block_loss.append(loss.detach().clone())
        block_correct.append((output.logits.detach().argmax(-1) == batch["labels"]).sum())
        block_examples += len(indices)
        if step % config["log_every"] == 0 or step == steps:
            record = {"step": step, "mean_training_loss": float(torch.stack(block_loss).mean()),
                      "training_accuracy": float(torch.stack(block_correct).sum()) / block_examples,
                      "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.monotonic() - loop_start}
            training.append(record)
            save_csv(args.output / "training.csv", training)
            print(f"{args.model} {args.mode} seed={args.seed} train_step={step} loss={record['mean_training_loss']:.4f} seconds={record['seconds']:.1f}", flush=True)
            block_loss, block_correct, block_examples = [], [], 0
        if step % config["eval_every"] == 0 or step == steps:
            record_selection(step)

    del train_model, optimizer
    gc.collect()
    checkpoint = torch.load(selected_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    scores, predicted, losses, norms = evaluate(model, data["evaluation"], config["eval_batch_size"], alpha_s, device, monitor=True)
    final_scales = collect_attention_scaling_stats(model)
    if args.mode == "bounded":
        validate_scale_records(final_scales, alpha_s, config["cap"])
    save_json(args.output / "selected_scale_stats.json", final_scales)
    save_csv(args.output / "evaluation_attention_inputs.csv", norms)
    save_csv(args.output / "predictions.csv", [{"example_id": int(i), "label": int(y), "prediction": int(p), "nll": float(nll)}
             for i, y, p, nll in zip(data["evaluation"]["example_id"].cpu(), data["evaluation"]["labels"].cpu(), predicted, losses)])
    result = {"model": model_config["model_id"], "name": args.model, "mode": args.mode, "seed": args.seed,
              "selected_step": best_step, "selection_accuracy": best_key[0], "evaluation": scores,
              "historical_baseline_target": model_config["baseline_target"],
              "historical_recovery_threshold": model_config["baseline_target"] - .01,
              "meets_historical_threshold": scores["accuracy"] >= model_config["baseline_target"] - .01,
              "alpha_s": alpha_s, "alpha_w": config["cap"], "configured_denominator_floor": alpha_s**2 * config["cap"]**2,
              "maximum_observed_matrix_to_alpha_ratio": max(r["maximum_matrix_to_alpha_ratio"] for r in norm_rows + norms),
              "applied_caps": applied_caps, "adapters": adapters,
              "seconds": time.monotonic() - start_time, "smoke": args.smoke,
              "checkpoint_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest()}
    save_json(args.output / "result.json", result)
    sources = [Path(__file__), ROOT / "experiments/lib/recovery_extension.py", ROOT / "experiments/lib/experiment.py",
               ROOT / "src/quantum_friendly_transformer/norm_transformer/attention_scaling.py",
               ROOT / "src/quantum_friendly_transformer/norm_transformer/frobenius_cap.py",
               ROOT / "src/quantum_friendly_transformer/experiments/runners/recovery/glue_sequence_classifier.py"]
    save_json(args.output / "manifest.json", manifest(config, sources, model=args.model, seed=args.seed, mode=args.mode,
              assets=json.loads((args.assets / "assets.json").read_text()), device=str(device),
              gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              dtype="float32 master parameters; bfloat16 autocast on CUDA", compiled=args.compile, smoke=args.smoke,
              limitations=["Historical scalar alpha floor, not a certified theorem input-matrix bound.",
                           "SST-2 validation is final evaluation; checkpoint selection uses only a split of training data.",
                           "Historical recovery targets are not a new matched native baseline or an equivalence test.",
                           "Normalization and architecture remain native; this does not validate the full theorem layer."]))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root', type=Path, help='Root for relative parent-checkpoint paths in the recipe')
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/configs/recovery/recovery_extension.json")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--model", choices=["1m", "8m", "33m"], required=True)
    parser.add_argument("--mode", choices=["bounded", "uniform"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
