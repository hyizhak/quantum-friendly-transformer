#!/usr/bin/env python
"""Sweep a binary classifier logit bias for a saved GLUE recovery checkpoint."""

from __future__ import annotations

import argparse
import json

from quantum_friendly_transformer.norm_transformer.frobenius_cap import (
    apply_learnable_frobenius_caps,
    collect_learnable_cap_stats,
)

from quantum_friendly_transformer.experiments.lib.common import ensure_parent, write_json
from quantum_friendly_transformer.experiments.runners.recovery.glue_distill import set_learnable_cap_max_by_weight_name
from quantum_friendly_transformer.experiments.runners.recovery.glue_sequence_classifier import (
    GLUE_TEXT_COLUMNS,
    call_model,
    load_model,
    make_loaders,
    pick_device,
    prepare_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=sorted(GLUE_TEXT_COLUMNS))
    parser.add_argument("--cap", type=float, required=True)
    parser.add_argument("--student-state-in", required=True)
    parser.add_argument("--save-student-state", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--pad-to-max-length", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--calibration-split", choices=["validation", "train"], default="validation")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--cap-norm-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--force-decoder-wrapper", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb-4bit-compute-dtype", choices=["bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    return parser.parse_args()


def clone_state_dict_cpu(model) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def classifier_bias_parameter(model):
    candidates = ["classifier.bias", "score.bias"]
    named = dict(model.named_parameters())
    for name in candidates:
        param = named.get(name)
        if param is not None and param.ndim == 1 and param.numel() == 2:
            return name, param
    for name, param in model.named_parameters():
        lowered = name.lower()
        if lowered.endswith("classifier.bias") or lowered.endswith("score.bias"):
            if param.ndim == 1 and param.numel() == 2:
                return name, param
    raise ValueError("could not find a binary classifier bias parameter")


def collect_logits(torch, model, eval_loader, device):
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for batch in eval_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = call_model(model, batch)
            logits.append(outputs.logits.detach().float().cpu())
            labels.append(batch["labels"].detach().cpu())
    return torch.cat(logits, dim=0), torch.cat(labels, dim=0)


def accuracy_for_delta(torch, margins, labels, delta: float) -> float:
    preds = (margins + float(delta) > 0).long()
    return float((preds == labels).float().mean().item())


def best_bias_delta(torch, logits, labels):
    if logits.shape[-1] != 2:
        raise ValueError(f"bias calibration only supports binary logits, got shape {tuple(logits.shape)}")
    margins = logits[:, 1] - logits[:, 0]
    thresholds = sorted(float(value) for value in (-margins).tolist())
    candidates = [0.0]
    if thresholds:
        candidates.append(thresholds[0] - 1.0)
        candidates.append(thresholds[-1] + 1.0)
        candidates.extend(thresholds)
        candidates.extend((left + right) / 2.0 for left, right in zip(thresholds, thresholds[1:]))
    best_delta = 0.0
    best_acc = accuracy_for_delta(torch, margins, labels, 0.0)
    for delta in candidates:
        acc = accuracy_for_delta(torch, margins, labels, delta)
        if acc > best_acc:
            best_acc = acc
            best_delta = float(delta)
    base_acc = accuracy_for_delta(torch, margins, labels, 0.0)
    return base_acc, best_delta, best_acc


def main() -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    args = parse_args()
    device = pick_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, local_files_only=args.offline)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    train_ds, eval_ds = prepare_dataset(args, tokenizer, load_dataset)
    train_loader, eval_loader = make_loaders(torch, args, tokenizer, train_ds, eval_ds)

    model = load_model(torch, args, tokenizer, device)
    apply_learnable_frobenius_caps(model, args.cap, norm_dtype=args.cap_norm_dtype)
    model.to(device)
    loaded_student_state = torch.load(args.student_state_in, map_location="cpu")
    cap_max_gamma_by_name = None
    if isinstance(loaded_student_state, dict) and "state_dict" in loaded_student_state:
        cap_max_gamma_by_name = loaded_student_state.get("cap_max_gamma_by_name")
        loaded_student_state = loaded_student_state["state_dict"]
    model.load_state_dict(loaded_student_state)
    if cap_max_gamma_by_name:
        set_learnable_cap_max_by_weight_name(model, cap_max_gamma_by_name, init_fraction=None)
    model.to(device)

    eval_logits, eval_labels = collect_logits(torch, model, eval_loader, device)
    if args.calibration_split == "train":
        calibration_logits, calibration_labels = collect_logits(torch, model, train_loader, device)
    else:
        calibration_logits, calibration_labels = eval_logits, eval_labels
    calibration_base_acc, delta, calibration_best_acc = best_bias_delta(
        torch,
        calibration_logits,
        calibration_labels,
    )
    eval_margins = eval_logits[:, 1] - eval_logits[:, 0]
    base_acc = accuracy_for_delta(torch, eval_margins, eval_labels, 0.0)
    calibrated_acc = accuracy_for_delta(torch, eval_margins, eval_labels, delta)
    bias_name, bias_param = classifier_bias_parameter(model)
    with torch.no_grad():
        bias_param[1].add_(float(delta))
    cap_stats = collect_learnable_cap_stats(model)
    n24 = sum(1 for row in cap_stats if abs(float(row["max_gamma"]) - 24.0) < 1e-5)
    n64 = sum(1 for row in cap_stats if abs(float(row["max_gamma"]) - 64.0) < 1e-5)

    payload = {
        "model": args.model,
        "task": args.task,
        "student_state_in": args.student_state_in,
        "bias_name": bias_name,
        "calibration_split": args.calibration_split,
        "class1_bias_delta": delta,
        "calibration_base_accuracy": calibration_base_acc,
        "calibration_best_accuracy": calibration_best_acc,
        "base_accuracy": base_acc,
        "calibrated_accuracy": calibrated_acc,
        "calibration_examples": int(calibration_labels.numel()),
        "eval_examples": int(eval_labels.numel()),
        "fn24": n24,
        "fn64": n64,
        "total_caps": len(cap_stats),
    }

    if args.save_student_state:
        state_output = ensure_parent(args.save_student_state)
        torch.save(
            {
                "state_dict": clone_state_dict_cpu(model),
                "cap_max_gamma_by_name": {row["name"]: row["max_gamma"] for row in cap_stats},
                "cap_stats": cap_stats,
            },
            state_output,
        )
        payload["saved_student_state"] = str(state_output)

    write_json(ensure_parent(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
