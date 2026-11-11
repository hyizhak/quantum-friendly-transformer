#!/usr/bin/env python
"""Train-split LayerNorm affine calibration for capped GLUE recovery states."""

from __future__ import annotations

import argparse
import json

from torch.nn.utils import parametrize

from quantum_friendly_transformer.norm_transformer.frobenius_cap import (
    apply_learnable_frobenius_caps,
    collect_learnable_cap_stats,
)

from quantum_friendly_transformer.experiments.lib.common import ensure_parent, seed_everything, write_json
from quantum_friendly_transformer.experiments.runners.recovery.glue_distill import (
    clone_student_state_with_caps,
    parse_layer_module_groups,
    set_learnable_cap_max_by_weight_name,
    set_learnable_cap_max_for_bert_layer_module_group,
)
from quantum_friendly_transformer.experiments.runners.recovery.glue_sequence_classifier import (
    GLUE_TEXT_COLUMNS,
    call_model,
    cap_summary,
    evaluate,
    load_model,
    make_loaders,
    meets_threshold,
    pick_device,
    prepare_dataset,
    primary_metric,
    threshold_value,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=sorted(GLUE_TEXT_COLUMNS))
    parser.add_argument("--cap", type=float, required=True)
    parser.add_argument("--student-state-in", required=True)
    parser.add_argument("--teacher-state-in", required=True)
    parser.add_argument("--layer-module-groups", nargs="+", required=True)
    parser.add_argument("--ln-scope", choices=["changed", "all"], default="changed")
    parser.add_argument("--calibration-batches", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--ln-blend", type=float, default=1.0)
    parser.add_argument("--gamma-clamp-min", type=float, default=None)
    parser.add_argument("--gamma-clamp-max", type=float, default=None)
    parser.add_argument("--beta-delta-clamp", type=float, default=None)
    parser.add_argument("--linear-bias-calibrate", action="store_true")
    parser.add_argument("--linear-bias-blend", type=float, default=1.0)
    parser.add_argument("--linear-bias-delta-clamp", type=float, default=None)
    parser.add_argument("--linear-weight-calibrate", action="store_true")
    parser.add_argument("--linear-weight-blend", type=float, default=1.0)
    parser.add_argument("--linear-weight-ridge", type=float, default=1e-3)
    parser.add_argument("--linear-weight-fit-bias", action="store_true")
    parser.add_argument("--baseline-target", type=float, default=None)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--pad-to-max-length", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="float32")
    parser.add_argument("--cap-norm-dtype", choices=["native", "float32"], default="float32")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--force-decoder-wrapper", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb-4bit-compute-dtype", choices=["bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-student-state", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_state_dict(torch, path: str):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"], state.get("cap_max_gamma_by_name")
    return state, None


def layernorm_modules(torch, model) -> dict[str, object]:
    return {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.LayerNorm)}


def changed_layernorm_names(groups: list[tuple[list[int], list[str]]]) -> set[str]:
    names: set[str] = set()
    attention_aliases = {"q", "k", "v", "attn_out"}
    ffn_aliases = {"intermediate", "ffn_out"}
    for layers, aliases in groups:
        alias_set = set(aliases)
        for layer in layers:
            if alias_set & attention_aliases:
                names.add(f"bert.encoder.layer.{layer}.attention.output.LayerNorm")
            if alias_set & ffn_aliases:
                names.add(f"bert.encoder.layer.{layer}.output.LayerNorm")
    return names



def changed_linear_names(groups: list[tuple[list[int], list[str]]]) -> set[str]:
    suffixes = {
        "q": "attention.self.query",
        "k": "attention.self.key",
        "v": "attention.self.value",
        "attn_out": "attention.output.dense",
        "intermediate": "intermediate.dense",
        "ffn_out": "output.dense",
    }
    names: set[str] = set()
    for layers, aliases in groups:
        for layer in layers:
            for alias in aliases:
                names.add(f"bert.encoder.layer.{layer}.{suffixes[alias]}")
    return names


def linear_modules(torch, model) -> dict[str, object]:
    return {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}



def effective_weight_cap(torch, module) -> float | None:
    if not parametrize.is_parametrized(module, "weight"):
        return None
    caps = []
    for parametrization in module.parametrizations.weight:
        gamma = getattr(parametrization, "gamma", None)
        if gamma is not None:
            caps.append(float(gamma.detach().float().cpu().item()))
    return min(caps) if caps else None


def assign_effective_weight(torch, module, weight, cap: float | None) -> tuple[float, float]:
    weight = weight.detach().float()
    original_norm = float(weight.norm().item())
    if cap is not None and original_norm > cap:
        weight = weight * (float(cap) / max(original_norm, 1e-12))
    projected_norm = float(weight.norm().item())
    with torch.no_grad():
        if parametrize.is_parametrized(module, "weight"):
            original = module.parametrizations.weight.original
            original.copy_(weight.to(device=original.device, dtype=original.dtype))
        else:
            module.weight.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
    return original_norm, projected_norm


def fit_linear_weights(
    torch,
    teacher,
    student,
    train_loader,
    device,
    selected_names: set[str],
    max_batches: int,
    ridge: float,
    blend: float,
    fit_bias: bool,
) -> dict:
    teacher_linears = linear_modules(torch, teacher)
    student_linears = linear_modules(torch, student)
    missing = sorted(name for name in selected_names if name not in student_linears or name not in teacher_linears)
    if missing:
        raise ValueError(f"selected Linear modules not found in both models: {missing}")

    stats = {}
    for name in sorted(selected_names):
        module = student_linears[name]
        in_features = int(module.weight.shape[1])
        out_features = int(module.weight.shape[0])
        use_bias = bool(fit_bias and module.bias is not None)
        width = in_features + (1 if use_bias else 0)
        stats[name] = {
            "n": 0,
            "xtx": torch.zeros((width, width), dtype=torch.float64),
            "xty": torch.zeros((width, out_features), dtype=torch.float64),
            "fit_bias": use_bias,
        }
    if not stats:
        return {"batches": 0, "linears": {}}

    teacher_cache = {}
    current_attention_mask = None
    handles = []

    def teacher_hook(name):
        def hook(_module, _inputs, output):
            teacher_cache[name] = output.detach().float()

        return hook

    def student_hook(name):
        def hook(_module, inputs, _output):
            target = teacher_cache.get(name)
            if target is None:
                return
            source = inputs[0].detach().float()
            x = source.reshape(-1, source.shape[-1])
            y = target.to(device=source.device, dtype=torch.float32).reshape(-1, target.shape[-1])
            if source.ndim == 3 and current_attention_mask is not None:
                mask = current_attention_mask
                if tuple(mask.shape[:2]) == tuple(source.shape[:2]):
                    keep = mask.reshape(-1).to(device=x.device, dtype=torch.bool)
                    x = x[keep]
                    y = y[keep]
            if x.numel() == 0:
                return
            row = stats[name]
            if row["fit_bias"]:
                ones = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
                x = torch.cat([x, ones], dim=1)
            x_cpu = x.double().cpu()
            y_cpu = y.double().cpu()
            row["n"] += int(x_cpu.shape[0])
            row["xtx"].add_(x_cpu.T.matmul(x_cpu))
            row["xty"].add_(x_cpu.T.matmul(y_cpu))

        return hook

    for name in stats:
        handles.append(teacher_linears[name].register_forward_hook(teacher_hook(name)))
        handles.append(student_linears[name].register_forward_hook(student_hook(name)))

    teacher.eval()
    student.eval()
    batches_seen = 0
    try:
        with torch.no_grad():
            for batch in train_loader:
                if batches_seen >= max_batches:
                    break
                batch = {key: value.to(device) for key, value in batch.items()}
                current_attention_mask = batch.get("attention_mask")
                teacher_cache.clear()
                call_model(teacher, batch)
                call_model(student, batch)
                teacher_cache.clear()
                batches_seen += 1
    finally:
        for handle in handles:
            handle.remove()

    fitted = {}
    with torch.no_grad():
        blend = min(max(float(blend), 0.0), 1.0)
        for name in sorted(stats):
            module = student_linears[name]
            row = stats[name]
            eye = torch.eye(row["xtx"].shape[0], dtype=torch.float64)
            coef = torch.linalg.solve(row["xtx"] + float(ridge) * eye, row["xty"])
            fitted_weight = coef[:-1].T if row["fit_bias"] else coef.T
            old_weight = module.weight.detach().float().cpu()
            new_weight = old_weight.double().lerp(fitted_weight, blend).float()
            cap = effective_weight_cap(torch, module)
            raw_norm, projected_norm = assign_effective_weight(torch, module, new_weight, cap)
            record = {
                "tokens": row["n"],
                "fit_bias": row["fit_bias"],
                "old_weight_norm": float(old_weight.norm().item()),
                "fit_weight_norm": float(fitted_weight.float().norm().item()),
                "blended_weight_norm": raw_norm,
                "projected_weight_norm": projected_norm,
                "cap": cap,
            }
            if row["fit_bias"]:
                fitted_bias = coef[-1]
                old_bias = module.bias.detach().double().cpu()
                new_bias = old_bias.lerp(fitted_bias, blend)
                module.bias.copy_(new_bias.to(device=module.bias.device, dtype=module.bias.dtype))
                record.update(
                    {
                        "old_bias_norm": float(old_bias.norm().item()),
                        "fit_bias_norm": float(fitted_bias.norm().item()),
                        "new_bias_norm": float(new_bias.norm().item()),
                    }
                )
            fitted[name] = record

    return {"batches": batches_seen, "linears": fitted}


def fit_linear_biases(
    torch,
    teacher,
    student,
    train_loader,
    device,
    selected_names: set[str],
    max_batches: int,
    blend: float,
    delta_clamp: float | None,
) -> dict:
    teacher_linears = linear_modules(torch, teacher)
    student_linears = linear_modules(torch, student)
    missing = sorted(name for name in selected_names if name not in student_linears or name not in teacher_linears)
    if missing:
        raise ValueError(f"selected Linear modules not found in both models: {missing}")

    stats = {}
    for name in sorted(selected_names):
        module = student_linears[name]
        if module.bias is None:
            continue
        stats[name] = {
            "n": 0,
            "sum_residual": torch.zeros(module.bias.numel(), dtype=torch.float64),
        }
    if not stats:
        return {"batches": 0, "linears": {}}

    teacher_cache = {}
    current_attention_mask = None
    handles = []

    def teacher_hook(name):
        def hook(_module, _inputs, output):
            teacher_cache[name] = output.detach().float()

        return hook

    def student_hook(name):
        def hook(_module, _inputs, output):
            target = teacher_cache.get(name)
            if target is None:
                return
            residual = target.to(device=output.device, dtype=torch.float32) - output.detach().float()
            residual = residual.reshape(-1, residual.shape[-1]).double()
            if output.ndim == 3 and current_attention_mask is not None:
                mask = current_attention_mask
                if tuple(mask.shape[:2]) == tuple(output.shape[:2]):
                    keep = mask.reshape(-1).to(device=residual.device, dtype=torch.bool)
                    residual = residual[keep]
            if residual.numel() == 0:
                return
            row = stats[name]
            row["n"] += int(residual.shape[0])
            row["sum_residual"].add_(residual.sum(dim=0).cpu())

        return hook

    for name in stats:
        handles.append(teacher_linears[name].register_forward_hook(teacher_hook(name)))
        handles.append(student_linears[name].register_forward_hook(student_hook(name)))

    teacher.eval()
    student.eval()
    batches_seen = 0
    try:
        with torch.no_grad():
            for batch in train_loader:
                if batches_seen >= max_batches:
                    break
                batch = {key: value.to(device) for key, value in batch.items()}
                current_attention_mask = batch.get("attention_mask")
                teacher_cache.clear()
                call_model(teacher, batch)
                call_model(student, batch)
                teacher_cache.clear()
                batches_seen += 1
    finally:
        for handle in handles:
            handle.remove()

    fitted = {}
    with torch.no_grad():
        blend = min(max(float(blend), 0.0), 1.0)
        for name in sorted(stats):
            module = student_linears[name]
            row = stats[name]
            n = max(1, row["n"])
            delta = row["sum_residual"] / n
            if delta_clamp is not None:
                delta = delta.clamp(-float(delta_clamp), float(delta_clamp))
            old_bias = module.bias.detach().double().cpu()
            new_bias = old_bias + blend * delta
            module.bias.copy_(new_bias.to(device=module.bias.device, dtype=module.bias.dtype))
            fitted[name] = {
                "tokens": n,
                "delta_min": float((new_bias - old_bias).min().item()),
                "delta_mean": float((new_bias - old_bias).mean().item()),
                "delta_max": float((new_bias - old_bias).max().item()),
            }

    return {"batches": batches_seen, "linears": fitted}


def fit_layernorm_affines(
    torch,
    teacher,
    student,
    train_loader,
    device,
    selected_names: set[str],
    max_batches: int,
    ridge: float,
    ln_blend: float,
    gamma_clamp_min: float | None,
    gamma_clamp_max: float | None,
    beta_delta_clamp: float | None,
) -> dict:
    teacher_lns = layernorm_modules(torch, teacher)
    student_lns = layernorm_modules(torch, student)
    missing = sorted(name for name in selected_names if name not in student_lns or name not in teacher_lns)
    if missing:
        raise ValueError(f"selected LayerNorm modules not found in both models: {missing}")

    stats = {}
    for name in sorted(selected_names):
        width = int(student_lns[name].normalized_shape[-1])
        stats[name] = {
            "n": 0,
            "sx": torch.zeros(width, dtype=torch.float64),
            "sy": torch.zeros(width, dtype=torch.float64),
            "sxx": torch.zeros(width, dtype=torch.float64),
            "sxy": torch.zeros(width, dtype=torch.float64),
        }

    teacher_cache = {}
    current_attention_mask = None
    handles = []

    def teacher_hook(name):
        def hook(_module, _inputs, output):
            teacher_cache[name] = output.detach().float()

        return hook

    def student_hook(name):
        def hook(module, inputs, _output):
            target = teacher_cache.get(name)
            if target is None:
                return
            source = inputs[0].detach().float()
            normalized = torch.nn.functional.layer_norm(
                source,
                module.normalized_shape,
                weight=None,
                bias=None,
                eps=module.eps,
            )
            x = normalized.reshape(-1, normalized.shape[-1]).double()
            y = target.to(device=source.device, dtype=torch.float32).reshape(-1, target.shape[-1]).double()
            if source.ndim == 3 and current_attention_mask is not None:
                mask = current_attention_mask
                if tuple(mask.shape[:2]) == tuple(source.shape[:2]):
                    keep = mask.reshape(-1).to(device=x.device, dtype=torch.bool)
                    x = x[keep]
                    y = y[keep]
            if x.numel() == 0:
                return
            row = stats[name]
            row["n"] += int(x.shape[0])
            row["sx"].add_(x.sum(dim=0).cpu())
            row["sy"].add_(y.sum(dim=0).cpu())
            row["sxx"].add_((x * x).sum(dim=0).cpu())
            row["sxy"].add_((x * y).sum(dim=0).cpu())

        return hook

    for name in selected_names:
        handles.append(teacher_lns[name].register_forward_hook(teacher_hook(name)))
        handles.append(student_lns[name].register_forward_hook(student_hook(name)))

    teacher.eval()
    student.eval()
    batches_seen = 0
    try:
        with torch.no_grad():
            for batch in train_loader:
                if batches_seen >= max_batches:
                    break
                batch = {key: value.to(device) for key, value in batch.items()}
                current_attention_mask = batch.get("attention_mask")
                teacher_cache.clear()
                call_model(teacher, batch)
                call_model(student, batch)
                teacher_cache.clear()
                batches_seen += 1
    finally:
        for handle in handles:
            handle.remove()

    fitted = {}
    with torch.no_grad():
        for name in sorted(selected_names):
            row = stats[name]
            n = max(1, row["n"])
            sx = row["sx"]
            sy = row["sy"]
            sxx = row["sxx"]
            sxy = row["sxy"]
            denom = sxx - (sx * sx / n)
            numer = sxy - (sx * sy / n)
            gamma = numer / (denom + float(ridge))
            beta = (sy - gamma * sx) / n
            module = student_lns[name]
            old_gamma = module.weight.detach().double().cpu()
            old_beta = module.bias.detach().double().cpu()
            if gamma_clamp_min is not None or gamma_clamp_max is not None:
                gamma = gamma.clamp(
                    min=float(gamma_clamp_min) if gamma_clamp_min is not None else None,
                    max=float(gamma_clamp_max) if gamma_clamp_max is not None else None,
                )
            if beta_delta_clamp is not None:
                delta = (beta - old_beta).clamp(-float(beta_delta_clamp), float(beta_delta_clamp))
                beta = old_beta + delta
            blend = min(max(float(ln_blend), 0.0), 1.0)
            if blend < 1.0:
                gamma = old_gamma.lerp(gamma, blend)
                beta = old_beta.lerp(beta, blend)
            module.weight.copy_(gamma.to(device=module.weight.device, dtype=module.weight.dtype))
            module.bias.copy_(beta.to(device=module.bias.device, dtype=module.bias.dtype))
            fitted[name] = {
                "tokens": n,
                "gamma_min": float(gamma.min().item()),
                "gamma_mean": float(gamma.mean().item()),
                "gamma_max": float(gamma.max().item()),
                "beta_min": float(beta.min().item()),
                "beta_mean": float(beta.mean().item()),
                "beta_max": float(beta.max().item()),
            }

    return {"batches": batches_seen, "layernorms": fitted}


def main() -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    args = parse_args()
    seed_everything(args.seed)
    metric = args.metric or primary_metric(args.task)
    device = pick_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, local_files_only=args.offline)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    train_ds, eval_ds = prepare_dataset(args, tokenizer, load_dataset)
    train_loader, eval_loader = make_loaders(torch, args, tokenizer, train_ds, eval_ds)

    teacher = load_model(torch, args, tokenizer, device)
    teacher_state, teacher_cap_max_gamma_by_name = load_state_dict(torch, args.teacher_state_in)
    if teacher_cap_max_gamma_by_name:
        apply_learnable_frobenius_caps(teacher, args.cap, norm_dtype=args.cap_norm_dtype)
    teacher.load_state_dict(teacher_state)
    if teacher_cap_max_gamma_by_name:
        set_learnable_cap_max_by_weight_name(teacher, teacher_cap_max_gamma_by_name, init_fraction=None)
    teacher.to(device)
    teacher.eval()

    student = load_model(torch, args, tokenizer, device)
    apply_learnable_frobenius_caps(student, args.cap, norm_dtype=args.cap_norm_dtype)
    student_state, cap_max_gamma_by_name = load_state_dict(torch, args.student_state_in)
    student.load_state_dict(student_state)
    if cap_max_gamma_by_name:
        set_learnable_cap_max_by_weight_name(student, cap_max_gamma_by_name, init_fraction=None)
    student.to(device)
    student.eval()

    base_metrics = evaluate(torch, student, eval_loader, device, args.task)
    groups = parse_layer_module_groups(args.layer_module_groups)
    lowered = []
    for spec, (layers, aliases) in zip(args.layer_module_groups, groups):
        updated = set_learnable_cap_max_for_bert_layer_module_group(
            student,
            args.cap,
            layers,
            aliases,
            init_fraction=None,
        )
        lowered.append({"spec": spec, "updated": updated})
    lowered_metrics = evaluate(torch, student, eval_loader, device, args.task)

    linear_weight_calibration = None
    linear_weight_metrics = None
    if args.linear_weight_calibrate:
        selected_linears = changed_linear_names(groups)
        linear_weight_calibration = fit_linear_weights(
            torch,
            teacher,
            student,
            train_loader,
            device,
            selected_linears,
            args.calibration_batches,
            args.linear_weight_ridge,
            args.linear_weight_blend,
            args.linear_weight_fit_bias,
        )
        linear_weight_metrics = evaluate(torch, student, eval_loader, device, args.task)

    linear_bias_calibration = None
    linear_bias_metrics = None
    if args.linear_bias_calibrate:
        selected_linears = changed_linear_names(groups)
        linear_bias_calibration = fit_linear_biases(
            torch,
            teacher,
            student,
            train_loader,
            device,
            selected_linears,
            args.calibration_batches,
            args.linear_bias_blend,
            args.linear_bias_delta_clamp,
        )
        linear_bias_metrics = evaluate(torch, student, eval_loader, device, args.task)

    selected_lns = set(layernorm_modules(torch, student)) if args.ln_scope == "all" else changed_layernorm_names(groups)
    calibration = fit_layernorm_affines(
        torch,
        teacher,
        student,
        train_loader,
        device,
        selected_lns,
        args.calibration_batches,
        args.ridge,
        args.ln_blend,
        args.gamma_clamp_min,
        args.gamma_clamp_max,
        args.beta_delta_clamp,
    )
    calibrated_metrics = evaluate(torch, student, eval_loader, device, args.task)
    cap_stats = collect_learnable_cap_stats(student)
    caps = {
        "fn24": sum(1 for row in cap_stats if abs(float(row["max_gamma"]) - 24.0) < 1e-5),
        "fn64": sum(1 for row in cap_stats if abs(float(row["max_gamma"]) - 64.0) < 1e-5),
        "total": len(cap_stats),
        "summary": cap_summary(cap_stats),
    }

    value = calibrated_metrics.get(metric)
    payload = {
        "model": args.model,
        "task": args.task,
        "metric": metric,
        "cap": args.cap,
        "student_state_in": args.student_state_in,
        "teacher_state_in": args.teacher_state_in,
        "layer_module_groups": args.layer_module_groups,
        "ln_scope": args.ln_scope,
        "calibration_batches": args.calibration_batches,
        "ridge": args.ridge,
        "ln_blend": args.ln_blend,
        "gamma_clamp_min": args.gamma_clamp_min,
        "gamma_clamp_max": args.gamma_clamp_max,
        "beta_delta_clamp": args.beta_delta_clamp,
        "linear_bias_calibrate": args.linear_bias_calibrate,
        "linear_bias_blend": args.linear_bias_blend,
        "linear_bias_delta_clamp": args.linear_bias_delta_clamp,
        "linear_weight_calibrate": args.linear_weight_calibrate,
        "linear_weight_blend": args.linear_weight_blend,
        "linear_weight_ridge": args.linear_weight_ridge,
        "linear_weight_fit_bias": args.linear_weight_fit_bias,
        "lowered": lowered,
        "base_metrics": base_metrics,
        "lowered_metrics": lowered_metrics,
        "linear_weight_metrics": linear_weight_metrics,
        "linear_bias_metrics": linear_bias_metrics,
        "calibrated_metrics": calibrated_metrics,
        "baseline_target": args.baseline_target,
        "threshold": threshold_value(args.baseline_target, metric, args.tolerance),
        "meets_threshold": meets_threshold(value, args.baseline_target, metric, args.tolerance),
        "caps": caps,
        "linear_weight_calibration": linear_weight_calibration,
        "linear_bias_calibration": linear_bias_calibration,
        "calibration": calibration,
    }

    if args.save_student_state:
        state_output = ensure_parent(args.save_student_state)
        torch.save(clone_student_state_with_caps(student), state_output)
        payload["saved_student_state"] = str(state_output)

    write_json(ensure_parent(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
