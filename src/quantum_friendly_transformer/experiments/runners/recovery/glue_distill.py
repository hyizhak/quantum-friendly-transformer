#!/usr/bin/env python
"""Warm-start GLUE recovery with teacher-logit distillation after a Frobenius cap."""

from __future__ import annotations

import argparse
import gc
import time

from quantum_friendly_transformer.norm_transformer.frobenius_cap import (
    apply_learnable_frobenius_caps,
    collect_learnable_cap_stats,
    target_weight_modules,
)

from quantum_friendly_transformer.experiments.lib.common import (
    ensure_parent,
    metadata_path,
    now_metadata,
    package_versions,
    seed_everything,
    write_json,
)
from quantum_friendly_transformer.experiments.runners.recovery.glue_sequence_classifier import (
    GLUE_TEXT_COLUMNS,
    TRAIN_SCOPE_CHOICES,
    cap_summary,
    call_model,
    configure_trainable,
    evaluate,
    fast_weight_stats,
    load_model,
    make_loaders,
    make_optimizer,
    meets_threshold,
    pick_device,
    prepare_dataset,
    primary_metric,
    select_baseline_target,
    threshold_value,
)
from quantum_friendly_transformer.experiments.runners.recovery.glue_warmstart import (
    cap_schedule,
    clone_state_dict_cpu,
    is_better,
    set_learnable_cap_max,
    stage_checkpoints,
    train_phase,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=sorted(GLUE_TEXT_COLUMNS))
    parser.add_argument("--cap", type=float, required=True)
    parser.add_argument("--anneal-caps", type=float, nargs="+", default=None)
    parser.add_argument("--anneal-stage-steps", type=int, nargs="+", default=None)
    parser.add_argument("--anneal-checkpoint-interval", type=int, default=0)
    parser.add_argument("--anneal-stage-learning-rates", type=float, nargs="+", default=None)
    parser.add_argument("--anneal-stage-ramp-to-next-cap", action="store_true")
    parser.add_argument(
        "--anneal-stage-ramp-preserve-lower-caps",
        action="store_true",
        help="During cap ramps, do not raise parametrizations whose current max cap is already below the ramp value.",
    )
    parser.add_argument(
        "--stage-lower-preserve-lower-caps",
        action="store_true",
        help="When moving to a lower anneal stage, do not raise parametrizations already below the new cap.",
    )
    parser.add_argument(
        "--anneal-stage-train-scopes",
        choices=TRAIN_SCOPE_CHOICES,
        nargs="+",
        default=None,
    )
    parser.add_argument("--baseline-checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--recovery-checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--baseline-target", type=float, default=None)
    parser.add_argument("--baseline-selection", choices=["final", "best"], default="best")
    parser.add_argument("--warm-start-selection", choices=["final", "best"], default="best")
    parser.add_argument("--anneal-selection", choices=["best", "final", "lookahead"], default="best")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--pad-to-max-length", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--baseline-learning-rate", type=float, default=2e-5)
    parser.add_argument("--recovery-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--classifier-lr-mult", type=float, default=1.0)
    parser.add_argument("--norm-lr-mult", type=float, default=1.0)
    parser.add_argument("--cap-scalar-lr-mult", type=float, default=1.0)
    parser.add_argument("--param-lr-mult-patterns", nargs="+", default=None)
    parser.add_argument("--param-lr-mults", type=float, nargs="+", default=None)
    parser.add_argument("--baseline-warmup-ratio", type=float, default=0.06)
    parser.add_argument("--recovery-warmup-ratio", type=float, default=0.03)
    parser.add_argument("--recovery-scheduler-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--baseline-train-scope", choices=TRAIN_SCOPE_CHOICES, default="all")
    parser.add_argument("--recovery-train-scope", choices=TRAIN_SCOPE_CHOICES, default="all")
    parser.add_argument(
        "--extra-trainable-patterns",
        nargs="+",
        default=None,
        help="Regexes for additional parameter names to train during distillation phases.",
    )
    parser.add_argument("--optimizer", choices=["sgd", "adamw", "adafactor"], default="adamw")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="float32")
    parser.add_argument("--cap-norm-dtype", choices=["native", "float32"], default="float32")
    parser.add_argument("--cap-init-fraction", type=float, default=0.999)
    parser.add_argument("--preserve-cap-fraction-on-lower", action="store_true")
    parser.add_argument("--freeze-cap-scalars", action="store_true")
    parser.add_argument("--student-dropout-p", type=float, default=None)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument("--ce-label-source", choices=["gold", "teacher"], default="gold")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--teacher-disagree-ce-multiplier", type=float, default=1.0)
    parser.add_argument("--class-weights", type=float, nargs="+", default=None)
    parser.add_argument("--label-margin-floor-weight", type=float, default=0.0)
    parser.add_argument("--label-margin-floor", type=float, default=1.0)
    parser.add_argument("--logit-mse-weight", type=float, default=0.0)
    parser.add_argument("--margin-distill-weight", type=float, default=0.0)
    parser.add_argument("--teacher-margin-floor-weight", type=float, default=0.0)
    parser.add_argument("--teacher-margin-floor", type=float, default=1.0)
    parser.add_argument("--hidden-distill-weight", type=float, default=0.0)
    parser.add_argument("--hidden-distill-layers", default="last")
    parser.add_argument("--hidden-distill-mode", choices=["cls", "mean"], default="cls")
    parser.add_argument("--hidden-distill-normalize", action="store_true")
    parser.add_argument("--attention-distill-weight", type=float, default=0.0)
    parser.add_argument("--attention-distill-layers", default="last")
    parser.add_argument("--rdrop-alpha", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--ema-start-step", type=int, default=0)
    parser.add_argument("--sam-rho", type=float, default=0.0)
    parser.add_argument("--sam-adaptive", action="store_true")
    parser.add_argument("--sam-eps", type=float, default=1e-12)
    parser.add_argument("--swa-start-step", type=int, default=None)
    parser.add_argument("--swa-every", type=int, default=1)
    parser.add_argument("--anneal-stage-hidden-distill-weights", type=float, nargs="+", default=None)
    parser.add_argument("--anneal-stage-logit-mse-weights", type=float, nargs="+", default=None)
    parser.add_argument("--anneal-stage-margin-distill-weights", type=float, nargs="+", default=None)
    parser.add_argument("--anneal-stage-attention-distill-weights", type=float, nargs="+", default=None)
    parser.add_argument("--anneal-stage-rdrop-alphas", type=float, nargs="+", default=None)
    parser.add_argument("--recovery-hidden-distill-weight", type=float, default=None)
    parser.add_argument("--recovery-logit-mse-weight", type=float, default=None)
    parser.add_argument("--recovery-margin-distill-weight", type=float, default=None)
    parser.add_argument("--recovery-attention-distill-weight", type=float, default=None)
    parser.add_argument("--recovery-rdrop-alpha", type=float, default=None)
    parser.add_argument("--layermodule-hidden-distill-weights", type=float, nargs="+", default=None)
    parser.add_argument("--layermodule-attention-distill-weights", type=float, nargs="+", default=None)
    parser.add_argument("--layermodule-rdrop-alphas", type=float, nargs="+", default=None)
    parser.add_argument("--layermodule-ln-gamma-mults", type=float, nargs="+", default=None)
    parser.add_argument("--layermodule-pre-ln-gamma-mults", type=float, nargs="+", default=None)
    parser.add_argument("--layermodule-pre-ln-beta-mults", type=float, nargs="+", default=None)
    parser.add_argument("--layermodule-linear-bias-mults", type=float, nargs="+", default=None)
    parser.add_argument("--anneal-stage-ce-weights", type=float, nargs="+", default=None)
    parser.add_argument("--recovery-ce-weight", type=float, default=None)
    parser.add_argument("--layerwise-final-groups", nargs="+", default=None)
    parser.add_argument("--layerwise-stage-steps", type=int, nargs="+", default=None)
    parser.add_argument("--layerwise-learning-rates", type=float, nargs="+", default=None)
    parser.add_argument("--layerwise-ce-weights", type=float, nargs="+", default=None)
    parser.add_argument("--layerwise-checkpoint-interval", type=int, default=None)
    parser.add_argument("--modulewise-final-groups", nargs="+", default=None)
    parser.add_argument("--modulewise-stage-steps", type=int, nargs="+", default=None)
    parser.add_argument("--modulewise-learning-rates", type=float, nargs="+", default=None)
    parser.add_argument("--modulewise-ce-weights", type=float, nargs="+", default=None)
    parser.add_argument("--modulewise-logit-mse-weights", type=float, nargs="+", default=None)
    parser.add_argument("--modulewise-margin-distill-weights", type=float, nargs="+", default=None)
    parser.add_argument("--modulewise-hidden-distill-weights", type=float, nargs="+", default=None)
    parser.add_argument("--modulewise-attention-distill-weights", type=float, nargs="+", default=None)
    parser.add_argument("--modulewise-rdrop-alphas", type=float, nargs="+", default=None)
    parser.add_argument("--modulewise-checkpoint-interval", type=int, default=None)
    parser.add_argument("--layermodule-final-groups", nargs="+", default=None)
    parser.add_argument("--layermodule-stage-steps", type=int, nargs="+", default=None)
    parser.add_argument("--layermodule-learning-rates", type=float, nargs="+", default=None)
    parser.add_argument("--layermodule-ce-weights", type=float, nargs="+", default=None)
    parser.add_argument("--layermodule-logit-mse-weights", type=float, nargs="+", default=None)
    parser.add_argument("--layermodule-margin-distill-weights", type=float, nargs="+", default=None)
    parser.add_argument("--layermodule-checkpoint-interval", type=int, default=None)
    parser.add_argument("--grouped-final-ramp", action="store_true")
    parser.add_argument("--grouped-ramp-fraction", type=float, default=1.0)
    parser.add_argument("--grouped-stage-selection", choices=["best", "final", "best_final_cap"], default="best")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--force-decoder-wrapper", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb-4bit-compute-dtype", choices=["bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--teacher-state-in", default=None)
    parser.add_argument("--save-teacher-state", default=None)
    parser.add_argument("--student-state-in", default=None)
    parser.add_argument("--save-student-state", default=None)
    parser.add_argument("--final-save-selection", choices=["none", "best", "final"], default="none")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def optimizer_args(args: argparse.Namespace, lr: float, train_scope: str) -> argparse.Namespace:
    return argparse.Namespace(
        optimizer=args.optimizer,
        learning_rate=lr,
        weight_decay=args.weight_decay,
        train_scope=train_scope,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
    )


def lr_multiplier_for_param(name: str, param, args: argparse.Namespace) -> float:
    import re

    lowered = name.lower()
    multiplier = 1.0
    is_classifier = lowered.startswith("score.") or ".score." in lowered or "classifier" in lowered
    is_norm = param.ndim == 1 and ("norm" in lowered or "ln_" in lowered)
    is_cap_scalar = "parametrizations.weight.0.g" in lowered
    if is_classifier:
        multiplier *= float(args.classifier_lr_mult)
    if is_norm:
        multiplier *= float(args.norm_lr_mult)
    if is_cap_scalar:
        multiplier *= float(args.cap_scalar_lr_mult)
    if args.param_lr_mult_patterns:
        for pattern, pattern_mult in zip(args.param_lr_mult_patterns, args.param_lr_mults or []):
            if re.search(pattern, name):
                multiplier *= float(pattern_mult)
    if multiplier <= 0.0:
        raise ValueError(f"learning-rate multiplier for {name} must be positive, got {multiplier}")
    return multiplier


def trainable_param_groups(model, args: argparse.Namespace, learning_rate: float) -> list[dict]:
    groups: dict[float, dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        multiplier = lr_multiplier_for_param(name, param, args)
        key = round(multiplier, 12)
        group = groups.setdefault(
            key,
            {
                "params": [],
                "lr": float(learning_rate) * multiplier,
                "lr_multiplier": multiplier,
            },
        )
        group["params"].append(param)
    return list(groups.values())


def sam_grad_norm(torch, params, adaptive: bool):
    norms = []
    shared_device = None
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if shared_device is None:
            shared_device = grad.device
        if adaptive:
            grad = grad * param.detach().abs()
        norms.append(torch.norm(grad, p=2).to(shared_device))
    if not norms:
        return None
    return torch.norm(torch.stack(norms), p=2)


def apply_sam_perturbation(torch, params, rho: float, adaptive: bool, eps: float):
    grad_norm = sam_grad_norm(torch, params, adaptive)
    if grad_norm is None:
        return [], None
    if not torch.isfinite(grad_norm) or grad_norm <= 0:
        return [], float(grad_norm.detach().cpu().item())
    perturbations = []
    scale = float(rho) / (grad_norm + float(eps))
    with torch.no_grad():
        for param in params:
            if param.grad is None:
                continue
            if adaptive:
                perturb = param.detach().abs().pow(2).mul(param.grad)
            else:
                perturb = param.grad
            perturb = perturb.mul(scale.to(device=param.device, dtype=param.dtype))
            param.add_(perturb)
            perturbations.append((param, perturb))
    return perturbations, float(grad_norm.detach().cpu().item())


def restore_sam_perturbation(torch, perturbations) -> None:
    with torch.no_grad():
        for param, perturb in perturbations:
            param.sub_(perturb)


def cap_lowering_init_fraction(args: argparse.Namespace) -> float | None:
    if args.preserve_cap_fraction_on_lower:
        return None
    return args.cap_init_fraction


def set_dropout_probability(model, p: float) -> int:
    import torch

    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability must be in [0, 1], got {p}")
    updated = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = float(p)
            updated += 1
    return updated


def parse_layer_group(spec: str) -> list[int]:
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"invalid layer range {part!r}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    if not layers:
        raise ValueError(f"empty layer group {spec!r}")
    return sorted(layers)


def parse_layer_groups(specs: list[str] | None) -> list[list[int]]:
    return [parse_layer_group(spec) for spec in (specs or [])]


def bert_layer_index(module_name: str) -> int | None:
    import re

    match = re.search(r"(?:^|\.)encoder\.layer\.(\d+)\.", module_name.lower())
    return int(match.group(1)) if match else None


def set_learnable_cap_max_for_bert_layers(
    model,
    max_gamma: float,
    layers: list[int] | set[int],
    init_fraction: float | None = None,
) -> int:
    import torch
    from torch.nn.utils import parametrize

    layer_set = set(layers)
    updated = 0
    for name, module in target_weight_modules(model):
        layer_index = bert_layer_index(name)
        if layer_index not in layer_set:
            continue
        if not parametrize.is_parametrized(module, "weight"):
            continue
        for parametrization in module.parametrizations.weight:
            if hasattr(parametrization, "max_gamma"):
                parametrization.max_gamma = float(max_gamma)
                if init_fraction is not None and hasattr(parametrization, "g"):
                    ratio = min(max(float(init_fraction), 1e-6), 1.0 - 1e-6)
                    value = torch.logit(torch.tensor(ratio, device=parametrization.g.device, dtype=parametrization.g.dtype))
                    parametrization.g.data.copy_(value)
                updated += 1
    return updated


MODULEWISE_ALIASES = {"q", "k", "v", "attn_out", "intermediate", "ffn_out"}


def parse_module_group(spec: str) -> list[str]:
    aliases = []
    for part in spec.split(","):
        alias = part.strip().lower()
        if not alias:
            continue
        if alias not in MODULEWISE_ALIASES:
            raise ValueError(f"unknown modulewise alias {alias!r}; expected one of {sorted(MODULEWISE_ALIASES)}")
        aliases.append(alias)
    if not aliases:
        raise ValueError(f"empty modulewise group {spec!r}")
    return aliases


def parse_module_groups(specs: list[str] | None) -> list[list[str]]:
    return [parse_module_group(spec) for spec in (specs or [])]


def parse_layer_module_groups(specs: list[str] | None) -> list[tuple[list[int], list[str]]]:
    groups = []
    for spec in specs or []:
        if ":" not in spec:
            raise ValueError(f"layermodule group {spec!r} must be formatted as layers:modules")
        layer_spec, module_spec = spec.split(":", 1)
        groups.append((parse_layer_group(layer_spec), parse_module_group(module_spec)))
    return groups


def module_name_matches_alias(module_name: str, alias: str) -> bool:
    lowered = module_name.lower()
    if alias == "q":
        return lowered.endswith("attention.self.query")
    if alias == "k":
        return lowered.endswith("attention.self.key")
    if alias == "v":
        return lowered.endswith("attention.self.value")
    if alias == "attn_out":
        return lowered.endswith("attention.output.dense")
    if alias == "intermediate":
        return lowered.endswith("intermediate.dense")
    if alias == "ffn_out":
        return lowered.endswith("output.dense") and not lowered.endswith("attention.output.dense")
    return False


def set_learnable_cap_max_for_bert_module_group(
    model,
    max_gamma: float,
    aliases: list[str] | set[str],
    init_fraction: float | None = None,
) -> int:
    import torch
    from torch.nn.utils import parametrize

    alias_set = set(aliases)
    updated = 0
    for name, module in target_weight_modules(model):
        if not any(module_name_matches_alias(name, alias) for alias in alias_set):
            continue
        if not parametrize.is_parametrized(module, "weight"):
            continue
        for parametrization in module.parametrizations.weight:
            if hasattr(parametrization, "max_gamma"):
                parametrization.max_gamma = float(max_gamma)
                if init_fraction is not None and hasattr(parametrization, "g"):
                    ratio = min(max(float(init_fraction), 1e-6), 1.0 - 1e-6)
                    value = torch.logit(torch.tensor(ratio, device=parametrization.g.device, dtype=parametrization.g.dtype))
                    parametrization.g.data.copy_(value)
                updated += 1
    return updated


def set_learnable_cap_max_for_bert_layer_module_group(
    model,
    max_gamma: float,
    layers: list[int] | set[int],
    aliases: list[str] | set[str],
    init_fraction: float | None = None,
) -> int:
    import torch
    from torch.nn.utils import parametrize

    layer_set = set(layers)
    alias_set = set(aliases)
    updated = 0
    for name, module in target_weight_modules(model):
        layer_index = bert_layer_index(name)
        if layer_index not in layer_set:
            continue
        if not any(module_name_matches_alias(name, alias) for alias in alias_set):
            continue
        if not parametrize.is_parametrized(module, "weight"):
            continue
        for parametrization in module.parametrizations.weight:
            if hasattr(parametrization, "max_gamma"):
                parametrization.max_gamma = float(max_gamma)
                if init_fraction is not None and hasattr(parametrization, "g"):
                    ratio = min(max(float(init_fraction), 1e-6), 1.0 - 1e-6)
                    value = torch.logit(torch.tensor(ratio, device=parametrization.g.device, dtype=parametrization.g.dtype))
                    parametrization.g.data.copy_(value)
                updated += 1
    return updated


def bert_linear_bias_suffix(alias: str) -> str | None:
    if alias == "q":
        return "attention.self.query.bias"
    if alias == "k":
        return "attention.self.key.bias"
    if alias == "v":
        return "attention.self.value.bias"
    if alias == "attn_out":
        return "attention.output.dense.bias"
    if alias == "intermediate":
        return "intermediate.dense.bias"
    if alias == "ffn_out":
        return "output.dense.bias"
    return None


def scale_bert_linear_bias_for_layer_module_group(
    model,
    layers: list[int] | set[int],
    aliases: list[str] | set[str],
    multiplier: float,
) -> list[str]:
    import torch

    if multiplier == 1.0:
        return []
    params = dict(model.named_parameters())
    updated = []
    with torch.no_grad():
        for layer in sorted(set(layers)):
            for alias in sorted(set(aliases)):
                suffix = bert_linear_bias_suffix(alias)
                if suffix is None:
                    continue
                name = f"bert.encoder.layer.{layer}.{suffix}"
                param = params.get(name)
                if param is None:
                    continue
                param.mul_(float(multiplier))
                updated.append(name)
    return updated


def scale_bert_layernorm_gamma_for_layer_module_group(
    model,
    layers: list[int] | set[int],
    aliases: list[str] | set[str],
    multiplier: float,
) -> list[str]:
    import torch

    if multiplier == 1.0:
        return []
    if multiplier <= 0.0:
        raise ValueError(f"LayerNorm gamma multiplier must be positive, got {multiplier}")
    alias_set = set(aliases)
    suffixes = []
    if alias_set.intersection({"q", "k", "v", "attn_out"}):
        suffixes.append("attention.output.LayerNorm.weight")
    if alias_set.intersection({"intermediate", "ffn_out"}):
        suffixes.append("output.LayerNorm.weight")
    params = dict(model.named_parameters())
    updated = []
    with torch.no_grad():
        for layer in sorted(set(layers)):
            for suffix in suffixes:
                name = f"bert.encoder.layer.{layer}.{suffix}"
                param = params.get(name)
                if param is None:
                    continue
                param.mul_(float(multiplier))
                updated.append(name)
    return updated


def scale_bert_pre_layernorm_for_layer_module_group(
    model,
    layers: list[int] | set[int],
    aliases: list[str] | set[str],
    gamma_multiplier: float = 1.0,
    beta_multiplier: float = 1.0,
) -> list[str]:
    import torch

    alias_set = set(aliases)
    if "intermediate" not in alias_set:
        return []
    if gamma_multiplier <= 0.0 or beta_multiplier <= 0.0:
        raise ValueError(
            f"pre-LayerNorm multipliers must be positive, got gamma={gamma_multiplier}, beta={beta_multiplier}"
        )
    params = dict(model.named_parameters())
    updated = []
    with torch.no_grad():
        for layer in sorted(set(layers)):
            gamma_name = f"bert.encoder.layer.{layer}.attention.output.LayerNorm.weight"
            gamma = params.get(gamma_name)
            if gamma is not None and gamma_multiplier != 1.0:
                gamma.mul_(float(gamma_multiplier))
                updated.append(gamma_name)
            beta_name = f"bert.encoder.layer.{layer}.attention.output.LayerNorm.bias"
            beta = params.get(beta_name)
            if beta is not None and beta_multiplier != 1.0:
                beta.mul_(float(beta_multiplier))
                updated.append(beta_name)
    return updated


def set_learnable_cap_max_by_weight_name(
    model,
    cap_max_gamma_by_name: dict[str, float],
    init_fraction: float | None = None,
) -> int:
    import torch
    from torch.nn.utils import parametrize

    updated = 0
    for name, module in target_weight_modules(model):
        weight_name = f"{name}.weight"
        if weight_name not in cap_max_gamma_by_name:
            continue
        if not parametrize.is_parametrized(module, "weight"):
            continue
        max_gamma = float(cap_max_gamma_by_name[weight_name])
        for parametrization in module.parametrizations.weight:
            if hasattr(parametrization, "max_gamma"):
                parametrization.max_gamma = max_gamma
                if init_fraction is not None and hasattr(parametrization, "g"):
                    ratio = min(max(float(init_fraction), 1e-6), 1.0 - 1e-6)
                    value = torch.logit(torch.tensor(ratio, device=parametrization.g.device, dtype=parametrization.g.dtype))
                    parametrization.g.data.copy_(value)
                updated += 1
    return updated


def set_learnable_cap_max_preserve_lower(
    model,
    max_gamma: float,
    init_fraction: float | None = None,
) -> int:
    import torch
    from torch.nn.utils import parametrize

    updated = 0
    for _, module in target_weight_modules(model):
        if not parametrize.is_parametrized(module, "weight"):
            continue
        for parametrization in module.parametrizations.weight:
            if not hasattr(parametrization, "max_gamma"):
                continue
            next_max_gamma = min(float(parametrization.max_gamma), float(max_gamma))
            parametrization.max_gamma = next_max_gamma
            if init_fraction is not None and hasattr(parametrization, "g"):
                ratio = min(max(float(init_fraction), 1e-6), 1.0 - 1e-6)
                value = torch.logit(torch.tensor(ratio, device=parametrization.g.device, dtype=parametrization.g.dtype))
                parametrization.g.data.copy_(value)
            updated += 1
    return updated


def clone_student_state_with_caps(model) -> dict:
    cap_stats = collect_learnable_cap_stats(model)
    return {
        "state_dict": clone_state_dict_cpu(model),
        "cap_max_gamma_by_name": {row["name"]: row["max_gamma"] for row in cap_stats},
        "cap_stats": cap_stats,
    }


def load_student_state_with_caps(model, state_snapshot) -> int:
    cap_max_gamma_by_name = None
    state_dict = state_snapshot
    if isinstance(state_snapshot, dict) and "state_dict" in state_snapshot:
        cap_max_gamma_by_name = state_snapshot.get("cap_max_gamma_by_name")
        state_dict = state_snapshot["state_dict"]
    model.load_state_dict(state_dict)
    if cap_max_gamma_by_name:
        return set_learnable_cap_max_by_weight_name(model, cap_max_gamma_by_name, init_fraction=None)
    return 0


def call_model_with_extra(model, batch, include_hidden: bool, include_attentions: bool):
    if not include_hidden and not include_attentions:
        return call_model(model, batch)
    extra_batch = dict(batch)
    if include_hidden:
        extra_batch["output_hidden_states"] = True
    if include_attentions:
        extra_batch["output_attentions"] = True
    return call_model(model, extra_batch)


def call_model_with_hidden(model, batch, include_hidden: bool):
    return call_model_with_extra(model, batch, include_hidden, False)


def hidden_layer_indices(spec: str, hidden_state_count: int) -> list[int]:
    selected: list[int] = []
    for raw_part in str(spec or "last").split(","):
        part = raw_part.strip().lower()
        if not part or part == "none":
            continue
        if part == "all":
            selected.extend(range(1, hidden_state_count))
            continue
        if part == "last":
            selected.append(hidden_state_count - 1)
            continue
        if part.startswith("last") and part[4:].isdigit():
            count = int(part[4:])
            selected.extend(range(max(1, hidden_state_count - count), hidden_state_count))
            continue
        index = int(part)
        if index < 0:
            index = hidden_state_count + index
        selected.append(index)
    unique: list[int] = []
    for index in selected:
        if index <= 0 or index >= hidden_state_count:
            raise ValueError(f"hidden layer index {index} out of range for {hidden_state_count} hidden states")
        if index not in unique:
            unique.append(index)
    return unique


def attention_layer_indices(spec: str, attention_count: int) -> list[int]:
    selected: list[int] = []
    for raw_part in str(spec or "last").split(","):
        part = raw_part.strip().lower()
        if not part or part == "none":
            continue
        if part == "all":
            selected.extend(range(attention_count))
            continue
        if part == "last":
            selected.append(attention_count - 1)
            continue
        if part.startswith("last") and part[4:].isdigit():
            count = int(part[4:])
            selected.extend(range(max(0, attention_count - count), attention_count))
            continue
        index = int(part)
        if index < 0:
            index = attention_count + index
        selected.append(index)
    unique: list[int] = []
    for index in selected:
        if index < 0 or index >= attention_count:
            raise ValueError(f"attention layer index {index} out of range for {attention_count} attention maps")
        if index not in unique:
            unique.append(index)
    return unique


def pooled_hidden(torch, hidden, attention_mask, mode: str):
    hidden = hidden.float()
    if mode == "cls":
        return hidden[:, 0]
    mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom


def hidden_distill_loss(torch, student_outputs, teacher_outputs, attention_mask, layers_spec: str, mode: str, normalize: bool):
    import torch.nn.functional as F

    student_hidden = getattr(student_outputs, "hidden_states", None)
    teacher_hidden = getattr(teacher_outputs, "hidden_states", None)
    if student_hidden is None or teacher_hidden is None:
        raise ValueError("hidden distillation requires outputs with hidden_states")
    if len(student_hidden) != len(teacher_hidden):
        raise ValueError(f"student/teacher hidden state count mismatch: {len(student_hidden)} vs {len(teacher_hidden)}")
    losses = []
    for index in hidden_layer_indices(layers_spec, len(student_hidden)):
        student_pooled = pooled_hidden(torch, student_hidden[index], attention_mask, mode)
        teacher_pooled = pooled_hidden(torch, teacher_hidden[index], attention_mask, mode).detach()
        if normalize:
            student_pooled = F.normalize(student_pooled, dim=-1)
            teacher_pooled = F.normalize(teacher_pooled, dim=-1)
        losses.append(F.mse_loss(student_pooled, teacher_pooled))
    if not losses:
        return None
    return sum(losses) / len(losses)


def attention_distill_loss(torch, student_outputs, teacher_outputs, attention_mask, layers_spec: str):
    student_attentions = getattr(student_outputs, "attentions", None)
    teacher_attentions = getattr(teacher_outputs, "attentions", None)
    if student_attentions is None or teacher_attentions is None:
        raise ValueError("attention distillation requires outputs with attentions")
    if len(student_attentions) != len(teacher_attentions):
        raise ValueError(
            f"student/teacher attention count mismatch: {len(student_attentions)} vs {len(teacher_attentions)}"
        )
    mask = attention_mask.float()
    pair_mask = mask[:, None, :, None] * mask[:, None, None, :]
    losses = []
    for index in attention_layer_indices(layers_spec, len(student_attentions)):
        student_attention = student_attentions[index].float()
        teacher_attention = teacher_attentions[index].detach().float()
        current_mask = pair_mask.to(device=student_attention.device, dtype=student_attention.dtype)
        denom = (current_mask.sum() * student_attention.shape[1]).clamp_min(1.0)
        losses.append(((student_attention - teacher_attention).pow(2) * current_mask).sum() / denom)
    if not losses:
        return None
    return sum(losses) / len(losses)


def symmetric_kl_loss(torch, logits_a, logits_b, temperature: float):
    import torch.nn.functional as F

    temperature = float(temperature)
    log_probs_a = F.log_softmax(logits_a / temperature, dim=-1)
    log_probs_b = F.log_softmax(logits_b / temperature, dim=-1)
    probs_a = log_probs_a.exp().detach()
    probs_b = log_probs_b.exp().detach()
    kl_ab = F.kl_div(log_probs_a, probs_b, reduction="batchmean")
    kl_ba = F.kl_div(log_probs_b, probs_a, reduction="batchmean")
    return 0.5 * (kl_ab + kl_ba) * (temperature * temperature)


def raw_logit_mse_loss(student_logits, teacher_logits):
    import torch.nn.functional as F

    return F.mse_loss(student_logits.float(), teacher_logits.detach().float())


def margin_distill_loss(student_logits, teacher_logits):
    import torch.nn.functional as F

    student_logits = student_logits.float()
    teacher_logits = teacher_logits.detach().float()
    if student_logits.shape[-1] == 2:
        student_margin = student_logits[:, 1] - student_logits[:, 0]
        teacher_margin = teacher_logits[:, 1] - teacher_logits[:, 0]
    else:
        student_margin = student_logits - student_logits.mean(dim=-1, keepdim=True)
        teacher_margin = teacher_logits - teacher_logits.mean(dim=-1, keepdim=True)
    return F.mse_loss(student_margin, teacher_margin)


def teacher_margin_floor_loss(student_logits, teacher_logits, margin_floor: float):
    import torch
    import torch.nn.functional as F

    student_logits = student_logits.float()
    teacher_logits = teacher_logits.detach().float()
    margin_floor = float(margin_floor)
    if student_logits.shape[-1] == 2:
        teacher_margin = teacher_logits[:, 1] - teacher_logits[:, 0]
        teacher_sign = torch.where(teacher_margin >= 0, 1.0, -1.0).to(student_logits.device)
        student_margin = student_logits[:, 1] - student_logits[:, 0]
        signed_student_margin = teacher_sign * student_margin
    else:
        teacher_pred = teacher_logits.argmax(dim=-1)
        student_pred_logit = student_logits.gather(1, teacher_pred[:, None]).squeeze(1)
        mask = torch.ones_like(student_logits, dtype=torch.bool)
        mask.scatter_(1, teacher_pred[:, None], False)
        student_other_max = student_logits.masked_fill(~mask, float("-inf")).max(dim=-1).values
        signed_student_margin = student_pred_logit - student_other_max
    return F.relu(margin_floor - signed_student_margin).pow(2).mean()


def label_margin_floor_loss(student_logits, labels, margin_floor: float):
    import torch
    import torch.nn.functional as F

    student_logits = student_logits.float()
    labels = labels.to(student_logits.device)
    margin_floor = float(margin_floor)
    if student_logits.shape[-1] == 2:
        label_sign = torch.where(labels == 1, 1.0, -1.0).to(student_logits.device)
        student_margin = student_logits[:, 1] - student_logits[:, 0]
        signed_student_margin = label_sign * student_margin
    else:
        label_logit = student_logits.gather(1, labels[:, None]).squeeze(1)
        mask = torch.ones_like(student_logits, dtype=torch.bool)
        mask.scatter_(1, labels[:, None], False)
        other_max = student_logits.masked_fill(~mask, float("-inf")).max(dim=-1).values
        signed_student_margin = label_logit - other_max
    return F.relu(margin_floor - signed_student_margin).pow(2).mean()


def distill_loss(
    torch,
    student_outputs,
    teacher_outputs,
    labels,
    attention_mask,
    args,
    ce_weight: float,
    logit_mse_weight: float,
    margin_weight: float,
    teacher_margin_floor_weight: float,
    hidden_weight: float,
    attention_weight: float,
):
    import torch.nn.functional as F

    temperature = float(args.distill_temperature)
    ce_weight = float(ce_weight)
    log_probs = F.log_softmax(student_outputs.logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_outputs.logits.detach() / temperature, dim=-1)
    kl = F.kl_div(log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)
    ce = None
    loss = kl
    if ce_weight > 0:
        if args.ce_label_source == "teacher":
            ce_labels = teacher_outputs.logits.detach().argmax(dim=-1)
        else:
            ce_labels = labels
        class_weights = None
        if args.class_weights is not None:
            if len(args.class_weights) != student_outputs.logits.shape[-1]:
                raise ValueError("--class-weights must match the number of labels")
            class_weights = torch.tensor(
                args.class_weights,
                device=student_outputs.logits.device,
                dtype=student_outputs.logits.dtype,
            )
        use_sample_weighted_ce = args.focal_gamma > 0 or args.teacher_disagree_ce_multiplier != 1.0
        if use_sample_weighted_ce:
            ce_per_sample = F.cross_entropy(
                student_outputs.logits,
                ce_labels,
                weight=class_weights,
                label_smoothing=float(args.label_smoothing),
                reduction="none",
            ).float()
            sample_weights = torch.ones_like(ce_per_sample)
            if args.focal_gamma > 0:
                label_probs = F.softmax(student_outputs.logits.float(), dim=-1)
                pt = label_probs.gather(1, ce_labels[:, None]).squeeze(1).clamp_min(1e-6)
                sample_weights = sample_weights * (1.0 - pt).pow(float(args.focal_gamma))
            if args.teacher_disagree_ce_multiplier != 1.0:
                teacher_pred = teacher_outputs.logits.detach().argmax(dim=-1)
                disagree = (teacher_pred != ce_labels).to(dtype=ce_per_sample.dtype)
                sample_weights = sample_weights * (1.0 + disagree * (float(args.teacher_disagree_ce_multiplier) - 1.0))
            ce = (sample_weights * ce_per_sample).sum() / sample_weights.sum().clamp_min(1e-6)
        else:
            ce = F.cross_entropy(
                student_outputs.logits,
                ce_labels,
                weight=class_weights,
                label_smoothing=float(args.label_smoothing),
            )
        loss = ce_weight * ce + (1.0 - ce_weight) * kl
    logit_mse = None
    logit_mse_weight = float(logit_mse_weight)
    if logit_mse_weight > 0:
        logit_mse = raw_logit_mse_loss(student_outputs.logits, teacher_outputs.logits)
        loss = loss + logit_mse_weight * logit_mse
    margin = None
    margin_weight = float(margin_weight)
    if margin_weight > 0:
        margin = margin_distill_loss(student_outputs.logits, teacher_outputs.logits)
        loss = loss + margin_weight * margin
    teacher_margin_floor = None
    teacher_margin_floor_weight = float(teacher_margin_floor_weight)
    if teacher_margin_floor_weight > 0:
        teacher_margin_floor = teacher_margin_floor_loss(
            student_outputs.logits, teacher_outputs.logits, args.teacher_margin_floor
        )
        loss = loss + teacher_margin_floor_weight * teacher_margin_floor
    if args.label_margin_floor_weight > 0:
        label_margin_floor = label_margin_floor_loss(
            student_outputs.logits, labels, args.label_margin_floor
        )
        loss = loss + float(args.label_margin_floor_weight) * label_margin_floor
    hidden = None
    hidden_weight = float(hidden_weight)
    if hidden_weight > 0:
        hidden = hidden_distill_loss(
            torch,
            student_outputs,
            teacher_outputs,
            attention_mask,
            args.hidden_distill_layers,
            args.hidden_distill_mode,
            args.hidden_distill_normalize,
        )
        if hidden is not None:
            loss = loss + hidden_weight * hidden
    attention = None
    attention_weight = float(attention_weight)
    if attention_weight > 0:
        attention = attention_distill_loss(
            torch,
            student_outputs,
            teacher_outputs,
            attention_mask,
            args.attention_distill_layers,
        )
        if attention is not None:
            loss = loss + attention_weight * attention
    return loss, ce, kl, logit_mse, margin, teacher_margin_floor, hidden, attention

def apply_extra_trainable_patterns(model, patterns: list[str] | None) -> tuple[int, list[str]]:
    if not patterns:
        return 0, []
    import re

    regexes = [re.compile(pattern) for pattern in patterns]
    matched = []
    total = 0
    for name, param in model.named_parameters():
        if any(regex.search(name) for regex in regexes):
            if not param.requires_grad:
                total += param.numel()
            param.requires_grad = True
            matched.append(name)
    if not matched:
        raise ValueError(f"--extra-trainable-patterns did not match any parameters: {patterns}")
    return total, matched


def train_distill_phase(
    torch,
    args: argparse.Namespace,
    student,
    teacher,
    train_loader,
    eval_loader,
    device,
    *,
    phase: str,
    cap: float,
    metric: str,
    target: float | None,
    checkpoints: list[int],
    learning_rate: float,
    train_scope: str,
    ce_weight: float | None = None,
    logit_mse_weight: float | None = None,
    margin_distill_weight: float | None = None,
    hidden_distill_weight: float | None = None,
    attention_distill_weight: float | None = None,
    rdrop_alpha: float | None = None,
    save_selection: str | None = None,
    lookahead_cap: float | None = None,
    ramp_start_cap: float | None = None,
    ramp_end_cap: float | None = None,
    ramp_steps: int | None = None,
    ramp_cap_setter=None,
) -> tuple[dict, dict | None]:
    from transformers import get_linear_schedule_with_warmup

    configure_trainable(student, train_scope, capped=True)
    extra_trainable_params, extra_trainable_names = apply_extra_trainable_patterns(
        student, args.extra_trainable_patterns
    )
    if args.freeze_cap_scalars:
        for name, param in student.named_parameters():
            if "parametrizations.weight.0.g" in name.lower():
                param.requires_grad = False
    trainable_params = [param for param in student.parameters() if param.requires_grad]
    trainable_groups = trainable_param_groups(student, args, learning_rate)
    optimizer = make_optimizer(optimizer_args(args, learning_rate, train_scope), trainable_groups)
    checkpoints = sorted(set(max(0, step) for step in checkpoints))
    max_steps = max(checkpoints)
    scheduler_total_steps = max_steps
    if args.recovery_scheduler_steps is not None:
        scheduler_total_steps = max(max_steps, int(args.recovery_scheduler_steps))
    warmup_steps = int(args.recovery_warmup_ratio * scheduler_total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, scheduler_total_steps) if max_steps > 0 else None

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    rows = []
    losses = []
    ce_losses = []
    kl_losses = []
    logit_mse_losses = []
    margin_losses = []
    teacher_margin_floor_losses = []
    hidden_losses = []
    attention_losses = []
    rdrop_losses = []
    sam_grad_norms = []
    start = time.time()
    selected_row = None
    selected_state = None
    selected_lookahead_row = None
    phase_ce_weight = args.ce_weight if ce_weight is None else ce_weight
    phase_logit_mse_weight = args.logit_mse_weight if logit_mse_weight is None else logit_mse_weight
    phase_margin_weight = args.margin_distill_weight if margin_distill_weight is None else margin_distill_weight
    phase_teacher_margin_floor_weight = args.teacher_margin_floor_weight
    phase_hidden_weight = args.hidden_distill_weight if hidden_distill_weight is None else hidden_distill_weight
    phase_attention_weight = (
        args.attention_distill_weight if attention_distill_weight is None else attention_distill_weight
    )
    phase_rdrop_alpha = args.rdrop_alpha if rdrop_alpha is None else rdrop_alpha
    ramp_enabled = ramp_start_cap is not None and ramp_end_cap is not None and max_steps > 0
    ramp_total_steps = max_steps if ramp_steps is None else max(0, int(ramp_steps))
    current_ramp_cap = None
    last_ramp_step = None
    ema_enabled = bool(getattr(args, "ema_decay", 0.0) and float(args.ema_decay) > 0)
    ema_decay = float(getattr(args, "ema_decay", 0.0))
    ema_start_step = max(0, int(getattr(args, "ema_start_step", 0)))
    ema_named_params = [(name, param) for name, param in student.named_parameters() if param.requires_grad]
    ema_params = None
    swa_enabled = args.swa_start_step is not None and int(args.swa_start_step) >= 0
    swa_start_step = 0 if args.swa_start_step is None else max(0, int(args.swa_start_step))
    swa_every = max(1, int(args.swa_every))
    swa_state = None
    swa_n = 0

    def ensure_ema_params():
        nonlocal ema_params
        if ema_params is None:
            ema_params = {name: param.detach().float().clone() for name, param in ema_named_params}

    def update_ema_params(step: int) -> None:
        if not ema_enabled or step < ema_start_step:
            return
        ensure_ema_params()
        with torch.no_grad():
            for name, param in ema_named_params:
                ema_params[name].mul_(ema_decay).add_(param.detach().float(), alpha=1.0 - ema_decay)

    def swap_to_ema():
        if not ema_enabled or ema_params is None:
            return None
        backup = {}
        with torch.no_grad():
            for name, param in ema_named_params:
                backup[name] = param.detach().clone()
                param.copy_(ema_params[name].to(device=param.device, dtype=param.dtype))
        return backup

    def restore_from_backup(backup) -> None:
        if backup is None:
            return
        with torch.no_grad():
            for name, param in ema_named_params:
                param.copy_(backup[name].to(device=param.device, dtype=param.dtype))

    def update_swa_state(step: int) -> None:
        nonlocal swa_state, swa_n
        if not swa_enabled or step < swa_start_step:
            return
        if (step - swa_start_step) % swa_every != 0:
            return
        current = student.state_dict()
        if swa_state is None:
            swa_state = {}
            for key, tensor in current.items():
                value = tensor.detach().cpu()
                if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
                    value = value.float()
                swa_state[key] = value.clone()
            swa_n = 1
            return
        next_n = swa_n + 1
        for key, tensor in current.items():
            value = tensor.detach().cpu()
            if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
                swa_state[key].add_(value.float().sub(swa_state[key]), alpha=1.0 / next_n)
            else:
                swa_state[key] = value.clone()
        swa_n = next_n

    def load_swa_state() -> None:
        if swa_state is None:
            return
        student.load_state_dict(swa_state)
        student.to(device)

    def ramp_cap_for_step(step: int) -> float | None:
        if not ramp_enabled:
            return None
        if ramp_total_steps <= 0:
            return float(ramp_end_cap)
        bounded_step = min(max(0, step), ramp_total_steps)
        fraction = bounded_step / ramp_total_steps
        return float(ramp_start_cap) + (float(ramp_end_cap) - float(ramp_start_cap)) * fraction

    def apply_ramp_cap(step: int) -> float | None:
        nonlocal current_ramp_cap, last_ramp_step
        cap_value = ramp_cap_for_step(step)
        if cap_value is None:
            return None
        bounded_step = min(max(0, step), max(0, ramp_total_steps))
        if last_ramp_step == bounded_step:
            return current_ramp_cap
        if ramp_cap_setter is None:
            if args.anneal_stage_ramp_preserve_lower_caps:
                set_learnable_cap_max_preserve_lower(student, cap_value, init_fraction=None)
            else:
                set_learnable_cap_max(student, cap_value, init_fraction=None)
        else:
            ramp_cap_setter(student, cap_value)
        current_ramp_cap = cap_value
        last_ramp_step = bounded_step
        return current_ramp_cap

    apply_ramp_cap(0)

    def evaluate_checkpoint(step: int):
        nonlocal selected_row, selected_state, selected_lookahead_row
        ramp_cap = apply_ramp_cap(step)
        metrics = evaluate(torch, student, eval_loader, device, args.task)
        value = metrics.get(metric)
        caps = collect_learnable_cap_stats(student)
        row = {
            "variant": f"fnlearn:{cap:g}:distill",
            "kind": "fnlearn_distill",
            "phase": phase,
            "cap": cap,
            "ramp_start_cap": ramp_start_cap,
            "ramp_end_cap": ramp_end_cap,
            "ramp_steps": ramp_total_steps if ramp_enabled else None,
            "ramp_cap": ramp_cap,
            "checkpoint_steps": step,
            "metric": metric,
            "metric_value": value,
            "baseline_target": target,
            "threshold": threshold_value(target, metric, args.tolerance),
            "meets_threshold": meets_threshold(value, target, metric, args.tolerance),
            "mean_train_loss": (sum(losses) / len(losses)) if losses else None,
            "mean_ce_loss": (sum(ce_losses) / len(ce_losses)) if ce_losses else None,
            "mean_kl_loss": (sum(kl_losses) / len(kl_losses)) if kl_losses else None,
            "mean_logit_mse_loss": (sum(logit_mse_losses) / len(logit_mse_losses)) if logit_mse_losses else None,
            "mean_margin_distill_loss": (sum(margin_losses) / len(margin_losses)) if margin_losses else None,
            "mean_teacher_margin_floor_loss": (
                sum(teacher_margin_floor_losses) / len(teacher_margin_floor_losses)
            ) if teacher_margin_floor_losses else None,
            "mean_hidden_loss": (sum(hidden_losses) / len(hidden_losses)) if hidden_losses else None,
            "mean_attention_loss": (sum(attention_losses) / len(attention_losses)) if attention_losses else None,
            "mean_rdrop_loss": (sum(rdrop_losses) / len(rdrop_losses)) if rdrop_losses else None,
            "mean_sam_grad_norm": (sum(sam_grad_norms) / len(sam_grad_norms)) if sam_grad_norms else None,
        "swa_start_step": swa_start_step if swa_enabled else None,
        "swa_every": swa_every if swa_enabled else None,
        "swa_n": swa_n if swa_enabled else None,
            "runtime_sec": time.time() - start,
            "metrics": metrics,
            "weight_stats_final": fast_weight_stats(student),
            "cap_stats": caps,
            "cap_summary": cap_summary(caps),
        }
        print_extra = ""
        ema_row = None
        if save_selection == "lookahead" and lookahead_cap is not None:
            checkpoint_state = clone_student_state_with_caps(student)
            updated = set_learnable_cap_max(student, lookahead_cap, init_fraction=cap_lowering_init_fraction(args))
            lookahead_metrics = evaluate(torch, student, eval_loader, device, args.task)
            lookahead_value = lookahead_metrics.get(metric)
            row["lookahead_cap"] = lookahead_cap
            row["lookahead_updated_parametrizations"] = updated
            row["lookahead_metric_value"] = lookahead_value
            row["lookahead_metrics"] = lookahead_metrics
            row["lookahead_meets_threshold"] = meets_threshold(lookahead_value, target, metric, args.tolerance)
            lookahead_row = dict(row)
            lookahead_row["metric_value"] = lookahead_value
            if is_better(metric, lookahead_row, selected_lookahead_row):
                selected_lookahead_row = lookahead_row
                selected_row = row
                selected_state = checkpoint_state
            load_student_state_with_caps(student, checkpoint_state)
            student.to(device)
            apply_ramp_cap(step)
            print_extra = f" lookahead_cap={lookahead_cap:g} lookahead_{metric}={lookahead_value:.6g}"
        rows.append(row)
        if save_selection == "best" and is_better(metric, row, selected_row):
            selected_row = row
            selected_state = clone_student_state_with_caps(student)
        elif save_selection == "best_final_cap":
            at_final_cap = not ramp_enabled or (
                ramp_cap is not None and abs(float(ramp_cap) - float(ramp_end_cap)) <= 1e-6
            )
            if at_final_cap and is_better(metric, row, selected_row):
                selected_row = row
                selected_state = clone_student_state_with_caps(student)
        elif save_selection == "final":
            selected_row = row
            selected_state = clone_student_state_with_caps(student)
        if ema_enabled and ema_params is not None and step >= ema_start_step:
            backup = swap_to_ema()
            ema_metrics = evaluate(torch, student, eval_loader, device, args.task)
            ema_value = ema_metrics.get(metric)
            ema_caps = collect_learnable_cap_stats(student)
            ema_row = dict(row)
            ema_row.update(
                {
                    "variant": f"fnlearn:{cap:g}:distill:ema",
                    "ema": True,
                    "ema_decay": ema_decay,
                    "ema_start_step": ema_start_step,
                    "metric_value": ema_value,
                    "metrics": ema_metrics,
                    "meets_threshold": meets_threshold(ema_value, target, metric, args.tolerance),
                    "weight_stats_final": fast_weight_stats(student),
                    "cap_stats": ema_caps,
                    "cap_summary": cap_summary(ema_caps),
                }
            )
            rows.append(ema_row)
            if save_selection == "best" and is_better(metric, ema_row, selected_row):
                selected_row = ema_row
                selected_state = clone_student_state_with_caps(student)
            elif save_selection == "best_final_cap":
                at_final_cap = not ramp_enabled or (
                    ramp_cap is not None and abs(float(ramp_cap) - float(ramp_end_cap)) <= 1e-6
                )
                if at_final_cap and is_better(metric, ema_row, selected_row):
                    selected_row = ema_row
                    selected_state = clone_student_state_with_caps(student)
            elif save_selection == "final":
                selected_row = ema_row
                selected_state = clone_student_state_with_caps(student)
            restore_from_backup(backup)
            print_extra += f" ema_{metric}={ema_value:.6g} ema_meets={meets_threshold(ema_value, target, metric, args.tolerance)}"
        if swa_enabled and swa_state is not None:
            backup_state = clone_student_state_with_caps(student)
            load_swa_state()
            swa_metrics = evaluate(torch, student, eval_loader, device, args.task)
            swa_value = swa_metrics.get(metric)
            swa_caps = collect_learnable_cap_stats(student)
            swa_row = dict(row)
            swa_row.update(
                {
                    "variant": f"fnlearn:{cap:g}:distill:swa",
                    "swa": True,
                    "swa_start_step": swa_start_step,
                    "swa_every": swa_every,
                    "swa_n": swa_n,
                    "metric_value": swa_value,
                    "metrics": swa_metrics,
                    "meets_threshold": meets_threshold(swa_value, target, metric, args.tolerance),
                    "weight_stats_final": fast_weight_stats(student),
                    "cap_stats": swa_caps,
                    "cap_summary": cap_summary(swa_caps),
                }
            )
            rows.append(swa_row)
            if save_selection == "best" and is_better(metric, swa_row, selected_row):
                selected_row = swa_row
                selected_state = clone_student_state_with_caps(student)
            elif save_selection == "best_final_cap":
                at_final_cap = not ramp_enabled or (
                    ramp_cap is not None and abs(float(ramp_cap) - float(ramp_end_cap)) <= 1e-6
                )
                if at_final_cap and is_better(metric, swa_row, selected_row):
                    selected_row = swa_row
                    selected_state = clone_student_state_with_caps(student)
            elif save_selection == "final":
                selected_row = swa_row
                selected_state = clone_student_state_with_caps(student)
            load_student_state_with_caps(student, backup_state)
            student.to(device)
            print_extra += f" swa_{metric}={swa_value:.6g} swa_n={swa_n} swa_meets={meets_threshold(swa_value, target, metric, args.tolerance)}"
        print(
            f"fnlearn:{cap:g}:distill {phase} step={step} {metric}={value:.6g} "
            f"threshold={threshold_value(target, metric, args.tolerance)} "
            f"meets={meets_threshold(value, target, metric, args.tolerance)}"
            f"{f' ramp_cap={ramp_cap:.6g}' if ramp_cap is not None else ''}"
            f"{print_extra}",
            flush=True,
        )

    if 0 in checkpoints:
        evaluate_checkpoint(0)

    sam_enabled = bool(getattr(args, "sam_rho", 0.0) and float(args.sam_rho) > 0.0)

    def compute_distill_batch(batch):
        include_hidden = phase_hidden_weight > 0
        include_attention = phase_attention_weight > 0
        with torch.no_grad():
            teacher_outputs = call_model_with_extra(teacher, batch, include_hidden, include_attention)
        outputs = call_model_with_extra(student, batch, include_hidden, include_attention)
        loss, ce, kl, logit_mse, margin, teacher_margin_floor, hidden, attention = distill_loss(
            torch,
            outputs,
            teacher_outputs,
            batch["labels"],
            batch["attention_mask"],
            args,
            phase_ce_weight,
            phase_logit_mse_weight,
            phase_margin_weight,
            phase_teacher_margin_floor_weight,
            phase_hidden_weight,
            phase_attention_weight,
        )
        rdrop = None
        if phase_rdrop_alpha and phase_rdrop_alpha > 0:
            outputs_b = call_model_with_extra(student, batch, include_hidden, include_attention)
            loss_b, ce_b, kl_b, logit_mse_b, margin_b, teacher_margin_floor_b, hidden_b, attention_b = distill_loss(
                torch,
                outputs_b,
                teacher_outputs,
                batch["labels"],
                batch["attention_mask"],
                args,
                phase_ce_weight,
                phase_logit_mse_weight,
                phase_margin_weight,
                phase_teacher_margin_floor_weight,
                phase_hidden_weight,
                phase_attention_weight,
            )
            rdrop = symmetric_kl_loss(torch, outputs.logits, outputs_b.logits, args.distill_temperature)
            loss = 0.5 * (loss + loss_b) + float(phase_rdrop_alpha) * rdrop
            if ce is not None and ce_b is not None:
                ce = 0.5 * (ce + ce_b)
            kl = 0.5 * (kl + kl_b)
            if logit_mse is not None and logit_mse_b is not None:
                logit_mse = 0.5 * (logit_mse + logit_mse_b)
            if margin is not None and margin_b is not None:
                margin = 0.5 * (margin + margin_b)
            if teacher_margin_floor is not None and teacher_margin_floor_b is not None:
                teacher_margin_floor = 0.5 * (teacher_margin_floor + teacher_margin_floor_b)
            if hidden is not None and hidden_b is not None:
                hidden = 0.5 * (hidden + hidden_b)
            if attention is not None and attention_b is not None:
                attention = 0.5 * (attention + attention_b)
        return loss, ce, kl, logit_mse, margin, teacher_margin_floor, hidden, attention, rdrop

    step_count = 0
    micro_step = 0
    accumulated_batches = []
    optimizer.zero_grad(set_to_none=True)
    student.train()
    while step_count < max_steps:
        for batch in train_loader:
            apply_ramp_cap(step_count)
            batch = {key: value.to(device) for key, value in batch.items()}
            if sam_enabled:
                accumulated_batches.append(batch)
            loss, ce, kl, logit_mse, margin, teacher_margin_floor, hidden, attention, rdrop = compute_distill_batch(batch)
            (loss / args.grad_accum_steps).backward()
            micro_step += 1
            if micro_step % args.grad_accum_steps != 0:
                continue
            sam_grad_norm = None
            if sam_enabled:
                perturbations, sam_grad_norm = apply_sam_perturbation(
                    torch,
                    trainable_params,
                    args.sam_rho,
                    args.sam_adaptive,
                    args.sam_eps,
                )
                optimizer.zero_grad(set_to_none=True)
                try:
                    if perturbations:
                        for sam_batch in accumulated_batches:
                            apply_ramp_cap(step_count)
                            loss, ce, kl, logit_mse, margin, teacher_margin_floor, hidden, attention, rdrop = compute_distill_batch(sam_batch)
                            (loss / args.grad_accum_steps).backward()
                finally:
                    restore_sam_perturbation(torch, perturbations)
                accumulated_batches.clear()
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu().item()))
            if ce is not None:
                ce_losses.append(float(ce.detach().cpu().item()))
            kl_losses.append(float(kl.detach().cpu().item()))
            if logit_mse is not None:
                logit_mse_losses.append(float(logit_mse.detach().cpu().item()))
            if margin is not None:
                margin_losses.append(float(margin.detach().cpu().item()))
            if teacher_margin_floor is not None:
                teacher_margin_floor_losses.append(float(teacher_margin_floor.detach().cpu().item()))
            if hidden is not None:
                hidden_losses.append(float(hidden.detach().cpu().item()))
            if attention is not None:
                attention_losses.append(float(attention.detach().cpu().item()))
            if rdrop is not None:
                rdrop_losses.append(float(rdrop.detach().cpu().item()))
            if sam_grad_norm is not None:
                sam_grad_norms.append(float(sam_grad_norm))
            step_count += 1
            update_ema_params(step_count)
            update_swa_state(step_count)
            apply_ramp_cap(step_count)
            if step_count in checkpoints:
                evaluate_checkpoint(step_count)
                student.train()
            if step_count >= max_steps:
                break

    result = {
        "variant": f"fnlearn:{cap:g}:distill",
        "kind": "fnlearn_distill",
        "phase": phase,
        "cap": cap,
        "trainable_params": sum(param.numel() for param in trainable_params),
        "extra_trainable_patterns": args.extra_trainable_patterns,
        "extra_trainable_params": extra_trainable_params,
        "extra_trainable_names": extra_trainable_names[:50],
        "learning_rate": learning_rate,
        "classifier_lr_mult": args.classifier_lr_mult,
        "norm_lr_mult": args.norm_lr_mult,
        "cap_scalar_lr_mult": args.cap_scalar_lr_mult,
        "param_lr_mult_patterns": args.param_lr_mult_patterns,
        "param_lr_mults": args.param_lr_mults,
        "warmup_ratio": args.recovery_warmup_ratio,
        "scheduler_total_steps": scheduler_total_steps,
        "scheduler_warmup_steps": warmup_steps,
        "train_scope": train_scope,
        "distill_temperature": args.distill_temperature,
        "ce_weight": phase_ce_weight,
        "ce_label_source": args.ce_label_source,
        "label_smoothing": args.label_smoothing,
        "focal_gamma": args.focal_gamma,
        "teacher_disagree_ce_multiplier": args.teacher_disagree_ce_multiplier,
        "class_weights": args.class_weights,
        "label_margin_floor_weight": args.label_margin_floor_weight,
        "label_margin_floor": args.label_margin_floor,
        "logit_mse_weight": phase_logit_mse_weight,
        "default_logit_mse_weight": args.logit_mse_weight,
        "margin_distill_weight": phase_margin_weight,
        "default_margin_distill_weight": args.margin_distill_weight,
        "teacher_margin_floor_weight": phase_teacher_margin_floor_weight,
        "teacher_margin_floor": args.teacher_margin_floor,
        "hidden_distill_weight": phase_hidden_weight,
        "default_hidden_distill_weight": args.hidden_distill_weight,
        "hidden_distill_layers": args.hidden_distill_layers,
        "hidden_distill_mode": args.hidden_distill_mode,
        "hidden_distill_normalize": args.hidden_distill_normalize,
        "attention_distill_weight": phase_attention_weight,
        "default_attention_distill_weight": args.attention_distill_weight,
        "attention_distill_layers": args.attention_distill_layers,
        "rdrop_alpha": phase_rdrop_alpha,
        "default_rdrop_alpha": args.rdrop_alpha,
        "ema_decay": ema_decay if ema_enabled else None,
        "ema_start_step": ema_start_step if ema_enabled else None,
        "sam_rho": args.sam_rho,
        "sam_adaptive": args.sam_adaptive,
        "sam_eps": args.sam_eps,
        "mean_sam_grad_norm": (sum(sam_grad_norms) / len(sam_grad_norms)) if sam_grad_norms else None,
        "swa_start_step": swa_start_step if swa_enabled else None,
        "swa_every": swa_every if swa_enabled else None,
        "swa_n": swa_n if swa_enabled else None,
        "ramp_start_cap": ramp_start_cap,
        "ramp_end_cap": ramp_end_cap,
        "ramp_steps": ramp_total_steps if ramp_enabled else None,
        "checkpoints": rows,
        "selected_checkpoint": selected_row,
        "save_selection": save_selection,
        "lookahead_cap": lookahead_cap,
    }
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, selected_state


def summarize_recovery(recovery_run: dict) -> dict:
    successes = [row for row in recovery_run["checkpoints"] if row.get("meets_threshold") is True]
    first = min(successes, key=lambda row: row["checkpoint_steps"]) if successes else None
    return {
        "recovered": first is not None,
        "minimal_recovery_steps": first["checkpoint_steps"] if first else None,
        "minimal_metric_value": first["metric_value"] if first else None,
    }


def main() -> None:
    args = parse_args()
    import os
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    seed_everything(args.seed)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True
    device = pick_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    train_ds, eval_ds = prepare_dataset(args, tokenizer, load_dataset)
    train_loader, eval_loader = make_loaders(torch, args, tokenizer, train_ds, eval_ds)

    metric = args.metric or primary_metric(args.task)
    if bool(args.param_lr_mult_patterns) != bool(args.param_lr_mults):
        raise ValueError("--param-lr-mult-patterns and --param-lr-mults must be provided together")
    if args.param_lr_mult_patterns is not None and len(args.param_lr_mult_patterns) != len(args.param_lr_mults):
        raise ValueError("--param-lr-mults must have one entry per --param-lr-mult-patterns item")
    schedule = cap_schedule(args)
    if args.anneal_stage_ce_weights is not None and len(args.anneal_stage_ce_weights) != max(0, len(schedule) - 1):
        raise ValueError("--anneal-stage-ce-weights must have one entry per non-final annealing cap")
    if args.anneal_stage_hidden_distill_weights is not None and len(args.anneal_stage_hidden_distill_weights) != max(0, len(schedule) - 1):
        raise ValueError("--anneal-stage-hidden-distill-weights must have one entry per non-final annealing cap")
    if args.anneal_stage_logit_mse_weights is not None and len(args.anneal_stage_logit_mse_weights) != max(0, len(schedule) - 1):
        raise ValueError("--anneal-stage-logit-mse-weights must have one entry per non-final annealing cap")
    if args.anneal_stage_margin_distill_weights is not None and len(args.anneal_stage_margin_distill_weights) != max(0, len(schedule) - 1):
        raise ValueError("--anneal-stage-margin-distill-weights must have one entry per non-final annealing cap")
    if args.anneal_stage_attention_distill_weights is not None and len(args.anneal_stage_attention_distill_weights) != max(0, len(schedule) - 1):
        raise ValueError("--anneal-stage-attention-distill-weights must have one entry per non-final annealing cap")
    if args.anneal_stage_rdrop_alphas is not None and len(args.anneal_stage_rdrop_alphas) != max(0, len(schedule) - 1):
        raise ValueError("--anneal-stage-rdrop-alphas must have one entry per non-final annealing cap")
    layer_groups = parse_layer_groups(args.layerwise_final_groups)
    module_groups = parse_module_groups(args.modulewise_final_groups)
    layer_module_groups = parse_layer_module_groups(args.layermodule_final_groups)
    grouped_modes = sum(bool(groups) for groups in (layer_groups, module_groups, layer_module_groups))
    if grouped_modes > 1:
        raise ValueError(
            "use only one grouped final mode: --layerwise-final-groups, "
            "--modulewise-final-groups, or --layermodule-final-groups"
        )
    if args.grouped_ramp_fraction < 0.0 or args.grouped_ramp_fraction > 1.0:
        raise ValueError("--grouped-ramp-fraction must be between 0.0 and 1.0")
    if layer_groups:
        expected_groups = len(layer_groups)
        if args.layerwise_stage_steps is None or len(args.layerwise_stage_steps) != expected_groups:
            raise ValueError("--layerwise-stage-steps must have one entry per layerwise final group")
        if args.layerwise_learning_rates is not None and len(args.layerwise_learning_rates) != expected_groups:
            raise ValueError("--layerwise-learning-rates must have one entry per layerwise final group")
        if args.layerwise_ce_weights is not None and len(args.layerwise_ce_weights) != expected_groups:
            raise ValueError("--layerwise-ce-weights must have one entry per layerwise final group")
    if module_groups:
        expected_groups = len(module_groups)
        if args.modulewise_stage_steps is None or len(args.modulewise_stage_steps) != expected_groups:
            raise ValueError("--modulewise-stage-steps must have one entry per modulewise final group")
        if args.modulewise_learning_rates is not None and len(args.modulewise_learning_rates) != expected_groups:
            raise ValueError("--modulewise-learning-rates must have one entry per modulewise final group")
        if args.modulewise_ce_weights is not None and len(args.modulewise_ce_weights) != expected_groups:
            raise ValueError("--modulewise-ce-weights must have one entry per modulewise final group")
        if args.modulewise_hidden_distill_weights is not None and len(args.modulewise_hidden_distill_weights) != expected_groups:
            raise ValueError("--modulewise-hidden-distill-weights must have one entry per modulewise final group")
        if args.modulewise_logit_mse_weights is not None and len(args.modulewise_logit_mse_weights) != expected_groups:
            raise ValueError("--modulewise-logit-mse-weights must have one entry per modulewise final group")
        if args.modulewise_margin_distill_weights is not None and len(args.modulewise_margin_distill_weights) != expected_groups:
            raise ValueError("--modulewise-margin-distill-weights must have one entry per modulewise final group")
        if args.modulewise_attention_distill_weights is not None and len(args.modulewise_attention_distill_weights) != expected_groups:
            raise ValueError("--modulewise-attention-distill-weights must have one entry per modulewise final group")
        if args.modulewise_rdrop_alphas is not None and len(args.modulewise_rdrop_alphas) != expected_groups:
            raise ValueError("--modulewise-rdrop-alphas must have one entry per modulewise final group")
    if layer_module_groups:
        expected_groups = len(layer_module_groups)
        if args.layermodule_stage_steps is None or len(args.layermodule_stage_steps) != expected_groups:
            raise ValueError("--layermodule-stage-steps must have one entry per layer-module final group")
        if args.layermodule_learning_rates is not None and len(args.layermodule_learning_rates) != expected_groups:
            raise ValueError("--layermodule-learning-rates must have one entry per layer-module final group")
        if args.layermodule_ce_weights is not None and len(args.layermodule_ce_weights) != expected_groups:
            raise ValueError("--layermodule-ce-weights must have one entry per layer-module final group")
        if args.layermodule_hidden_distill_weights is not None and len(args.layermodule_hidden_distill_weights) != expected_groups:
            raise ValueError("--layermodule-hidden-distill-weights must have one entry per layer-module final group")
        if args.layermodule_logit_mse_weights is not None and len(args.layermodule_logit_mse_weights) != expected_groups:
            raise ValueError("--layermodule-logit-mse-weights must have one entry per layer-module final group")
        if args.layermodule_margin_distill_weights is not None and len(args.layermodule_margin_distill_weights) != expected_groups:
            raise ValueError("--layermodule-margin-distill-weights must have one entry per layer-module final group")
        if args.layermodule_attention_distill_weights is not None and len(args.layermodule_attention_distill_weights) != expected_groups:
            raise ValueError("--layermodule-attention-distill-weights must have one entry per layer-module final group")
        if args.layermodule_rdrop_alphas is not None and len(args.layermodule_rdrop_alphas) != expected_groups:
            raise ValueError("--layermodule-rdrop-alphas must have one entry per layer-module final group")
        if args.layermodule_ln_gamma_mults is not None and len(args.layermodule_ln_gamma_mults) != expected_groups:
            raise ValueError("--layermodule-ln-gamma-mults must have one entry per layer-module final group")
        if args.layermodule_pre_ln_gamma_mults is not None and len(args.layermodule_pre_ln_gamma_mults) != expected_groups:
            raise ValueError("--layermodule-pre-ln-gamma-mults must have one entry per layer-module final group")
        if args.layermodule_pre_ln_beta_mults is not None and len(args.layermodule_pre_ln_beta_mults) != expected_groups:
            raise ValueError("--layermodule-pre-ln-beta-mults must have one entry per layer-module final group")
        if args.layermodule_linear_bias_mults is not None and len(args.layermodule_linear_bias_mults) != expected_groups:
            raise ValueError("--layermodule-linear-bias-mults must have one entry per layer-module final group")
    layerwise_checkpoint_interval = (
        args.anneal_checkpoint_interval
        if args.layerwise_checkpoint_interval is None
        else args.layerwise_checkpoint_interval
    )
    modulewise_checkpoint_interval = (
        args.anneal_checkpoint_interval
        if args.modulewise_checkpoint_interval is None
        else args.modulewise_checkpoint_interval
    )
    layermodule_checkpoint_interval = (
        args.anneal_checkpoint_interval
        if args.layermodule_checkpoint_interval is None
        else args.layermodule_checkpoint_interval
    )

    def grouped_ramp_steps(group_checkpoints: list[int]) -> int | None:
        if not args.grouped_final_ramp or not group_checkpoints:
            return None
        max_step = max(group_checkpoints)
        return min(max_step, max(0, int(round(max_step * args.grouped_ramp_fraction))))

    teacher = load_model(torch, args, tokenizer, device)
    if args.teacher_state_in:
        print(f"loading distillation teacher baseline from {args.teacher_state_in}", flush=True)
        loaded_teacher_state = torch.load(args.teacher_state_in, map_location="cpu")
        loaded_teacher_cap_max_gamma_by_name = None
        if isinstance(loaded_teacher_state, dict) and "state_dict" in loaded_teacher_state:
            loaded_teacher_cap_max_gamma_by_name = loaded_teacher_state.get("cap_max_gamma_by_name")
            loaded_teacher_state = loaded_teacher_state["state_dict"]
        if loaded_teacher_cap_max_gamma_by_name:
            teacher_cap = max(float(value) for value in loaded_teacher_cap_max_gamma_by_name.values())
            print(f"applying learnable FN caps to capped distillation teacher at max {teacher_cap:g}", flush=True)
            apply_learnable_frobenius_caps(
                teacher,
                teacher_cap,
                init_fraction=args.cap_init_fraction,
                norm_dtype=args.cap_norm_dtype,
            )
            updated = set_learnable_cap_max_by_weight_name(
                teacher, loaded_teacher_cap_max_gamma_by_name, init_fraction=None
            )
            print(f"restored mixed FN cap maxima on teacher ({updated} parametrizations)", flush=True)
        teacher.load_state_dict(loaded_teacher_state)
        teacher.to(device)
        metrics = evaluate(torch, teacher, eval_loader, device, args.task)
        value = metrics.get(metric)
        baseline_run = {
            "variant": "baseline",
            "kind": "baseline",
            "phase": "baseline_loaded",
            "checkpoints": [
                {
                    "variant": "baseline",
                    "kind": "baseline",
                    "phase": "baseline_loaded",
                    "checkpoint_steps": 0,
                    "metric": metric,
                    "metric_value": value,
                    "metrics": metrics,
                }
            ],
            "selected_checkpoint": {
                "checkpoint_steps": 0,
                "metric": metric,
                "metric_value": value,
                "metrics": metrics,
            },
            "save_selection": "loaded",
        }
        selected_state = clone_state_dict_cpu(teacher)
    else:
        print(f"running distillation teacher baseline on {args.task}", flush=True)
        baseline_run, selected_state = train_phase(
            torch,
            args,
            teacher,
            train_loader,
            eval_loader,
            device,
            phase="baseline",
            cap=None,
            metric=metric,
            target=None,
            checkpoints=args.baseline_checkpoints,
            learning_rate=args.baseline_learning_rate,
            warmup_ratio=args.baseline_warmup_ratio,
            train_scope=args.baseline_train_scope,
            save_selection=args.warm_start_selection,
        )
        if selected_state is None:
            selected_state = clone_state_dict_cpu(teacher)
        teacher.load_state_dict(selected_state)
        teacher.to(device)
    if args.save_teacher_state:
        teacher_state_output = ensure_parent(args.save_teacher_state)
        torch.save(selected_state, teacher_state_output)
        print(f"saved teacher state {teacher_state_output}", flush=True)
    target = args.baseline_target
    if target is None:
        target = select_baseline_target(baseline_run["checkpoints"], metric, args.baseline_selection)

    student = load_model(torch, args, tokenizer, device)
    selected_state_has_parametrizations = any(
        isinstance(key, str) and ".parametrizations." in key for key in selected_state.keys()
    )
    if selected_state_has_parametrizations and args.student_state_in:
        print(
            "skipping pre-cap student load from parametrized teacher; "
            "will load --student-state-in after applying caps",
            flush=True,
        )
    else:
        student.load_state_dict(selected_state)
    student.to(device)
    weight_stats_before = fast_weight_stats(student)
    print(f"applying learnable FN cap {schedule[0]:g} to distillation student", flush=True)
    applied_caps = apply_learnable_frobenius_caps(
        student,
        schedule[0],
        init_fraction=args.cap_init_fraction,
        norm_dtype=args.cap_norm_dtype,
    )
    student.to(device)
    if args.student_dropout_p is not None:
        updated_dropouts = set_dropout_probability(student, args.student_dropout_p)
        print(f"set student dropout p={args.student_dropout_p:g} on {updated_dropouts} modules", flush=True)
    if args.student_state_in:
        print(f"loading capped student state from {args.student_state_in}", flush=True)
        loaded_student_state = torch.load(args.student_state_in, map_location="cpu")
        loaded_cap_max_gamma_by_name = None
        if isinstance(loaded_student_state, dict) and "state_dict" in loaded_student_state:
            loaded_cap_max_gamma_by_name = loaded_student_state.get("cap_max_gamma_by_name")
            loaded_student_state = loaded_student_state["state_dict"]
        student.load_state_dict(loaded_student_state)
        if loaded_cap_max_gamma_by_name:
            updated = set_learnable_cap_max_by_weight_name(student, loaded_cap_max_gamma_by_name, init_fraction=None)
            print(f"restored mixed FN cap maxima on {updated} parametrizations", flush=True)
        student.to(device)
    weight_stats_after_cap = fast_weight_stats(student)

    final_save_selection = None if args.final_save_selection == "none" else args.final_save_selection
    final_selected_state = None
    last_selected_state = None
    recovery_runs = []
    interrupted = False
    try:
        for stage_index, stage_cap in enumerate(schedule):
            previous_stage_cap = schedule[stage_index - 1] if stage_index > 0 else stage_cap
            final_stage = stage_index == len(schedule) - 1
            grouped_final_stage = final_stage and (bool(layer_groups) or bool(module_groups) or bool(layer_module_groups))
            if stage_index > 0:
                if grouped_final_stage:
                    print(f"entering grouped final FN cap {stage_cap:g}; keeping previous caps before grouped lowering", flush=True)
                elif args.anneal_stage_ramp_to_next_cap:
                    if args.stage_lower_preserve_lower_caps:
                        updated = set_learnable_cap_max_preserve_lower(student, stage_cap, init_fraction=None)
                    else:
                        updated = set_learnable_cap_max(student, stage_cap, init_fraction=None)
                    print(f"set learnable FN cap to {stage_cap:g} after ramp on {updated} parametrizations", flush=True)
                else:
                    if args.stage_lower_preserve_lower_caps:
                        updated = set_learnable_cap_max_preserve_lower(
                            student, stage_cap, init_fraction=cap_lowering_init_fraction(args)
                        )
                    else:
                        updated = set_learnable_cap_max(student, stage_cap, init_fraction=cap_lowering_init_fraction(args))
                    print(f"lowered learnable FN cap to {stage_cap:g} on {updated} parametrizations", flush=True)
            if final_stage:
                checkpoints = args.recovery_checkpoints
                phase = "distill_recovery"
                stage_learning_rate = args.recovery_learning_rate
                stage_ce_weight = args.recovery_ce_weight if args.recovery_ce_weight is not None else args.ce_weight
                stage_logit_mse_weight = (
                    args.recovery_logit_mse_weight
                    if args.recovery_logit_mse_weight is not None
                    else args.logit_mse_weight
                )
                stage_margin_weight = (
                    args.recovery_margin_distill_weight
                    if args.recovery_margin_distill_weight is not None
                    else args.margin_distill_weight
                )
                stage_hidden_weight = (
                    args.recovery_hidden_distill_weight
                    if args.recovery_hidden_distill_weight is not None
                    else args.hidden_distill_weight
                )
                stage_attention_weight = (
                    args.recovery_attention_distill_weight
                    if args.recovery_attention_distill_weight is not None
                    else args.attention_distill_weight
                )
                stage_rdrop_alpha = (
                    args.recovery_rdrop_alpha
                    if args.recovery_rdrop_alpha is not None
                    else args.rdrop_alpha
                )
            else:
                checkpoints = stage_checkpoints(args.anneal_stage_steps[stage_index], args.anneal_checkpoint_interval)
                phase = f"distill_anneal_cap_{stage_cap:g}"
                stage_learning_rate = (
                    args.anneal_stage_learning_rates[stage_index]
                    if args.anneal_stage_learning_rates is not None
                    else args.recovery_learning_rate
                )
                stage_ce_weight = (
                    args.anneal_stage_ce_weights[stage_index]
                    if args.anneal_stage_ce_weights is not None
                    else args.ce_weight
                )
                stage_logit_mse_weight = (
                    args.anneal_stage_logit_mse_weights[stage_index]
                    if args.anneal_stage_logit_mse_weights is not None
                    else args.logit_mse_weight
                )
                stage_margin_weight = (
                    args.anneal_stage_margin_distill_weights[stage_index]
                    if args.anneal_stage_margin_distill_weights is not None
                    else args.margin_distill_weight
                )
                stage_hidden_weight = (
                    args.anneal_stage_hidden_distill_weights[stage_index]
                    if args.anneal_stage_hidden_distill_weights is not None
                    else args.hidden_distill_weight
                )
                stage_attention_weight = (
                    args.anneal_stage_attention_distill_weights[stage_index]
                    if args.anneal_stage_attention_distill_weights is not None
                    else args.attention_distill_weight
                )
                stage_rdrop_alpha = (
                    args.anneal_stage_rdrop_alphas[stage_index]
                    if args.anneal_stage_rdrop_alphas is not None
                    else args.rdrop_alpha
                )
            if grouped_final_stage and layer_module_groups:
                for group_index, (group_spec, (layers, aliases)) in enumerate(zip(args.layermodule_final_groups, layer_module_groups)):
                    if args.grouped_final_ramp:
                        updated = set_learnable_cap_max_for_bert_layer_module_group(
                            student,
                            previous_stage_cap,
                            layers,
                            aliases,
                            init_fraction=None,
                        )
                        print(
                            f"ramping layer-module FN cap from {previous_stage_cap:g} to {stage_cap:g} "
                            f"on BERT {group_spec} ({updated} parametrizations)",
                            flush=True,
                        )
                    else:
                        updated = set_learnable_cap_max_for_bert_layer_module_group(
                            student,
                            stage_cap,
                            layers,
                            aliases,
                            init_fraction=cap_lowering_init_fraction(args),
                        )
                        print(
                            f"lowered layer-module FN cap to {stage_cap:g} on BERT {group_spec} "
                            f"({updated} parametrizations)",
                            flush=True,
                        )
                    group_checkpoints = stage_checkpoints(args.layermodule_stage_steps[group_index], layermodule_checkpoint_interval)
                    group_learning_rate = (
                        args.layermodule_learning_rates[group_index]
                        if args.layermodule_learning_rates is not None
                        else args.recovery_learning_rate
                    )
                    group_ce_weight = (
                        args.layermodule_ce_weights[group_index]
                        if args.layermodule_ce_weights is not None
                        else stage_ce_weight
                    )
                    group_logit_mse_weight = (
                        args.layermodule_logit_mse_weights[group_index]
                        if args.layermodule_logit_mse_weights is not None
                        else stage_logit_mse_weight
                    )
                    group_margin_weight = (
                        args.layermodule_margin_distill_weights[group_index]
                        if args.layermodule_margin_distill_weights is not None
                        else stage_margin_weight
                    )
                    group_hidden_weight = (
                        args.layermodule_hidden_distill_weights[group_index]
                        if args.layermodule_hidden_distill_weights is not None
                        else stage_hidden_weight
                    )
                    group_attention_weight = (
                        args.layermodule_attention_distill_weights[group_index]
                        if args.layermodule_attention_distill_weights is not None
                        else stage_attention_weight
                    )
                    group_rdrop_alpha = (
                        args.layermodule_rdrop_alphas[group_index]
                        if args.layermodule_rdrop_alphas is not None
                        else stage_rdrop_alpha
                    )
                    group_pre_ln_gamma_mult = (
                        args.layermodule_pre_ln_gamma_mults[group_index]
                        if args.layermodule_pre_ln_gamma_mults is not None
                        else 1.0
                    )
                    group_pre_ln_beta_mult = (
                        args.layermodule_pre_ln_beta_mults[group_index]
                        if args.layermodule_pre_ln_beta_mults is not None
                        else group_pre_ln_gamma_mult
                    )
                    if group_pre_ln_gamma_mult != 1.0 or group_pre_ln_beta_mult != 1.0:
                        updated_pre_lns = scale_bert_pre_layernorm_for_layer_module_group(
                            student, layers, aliases, group_pre_ln_gamma_mult, group_pre_ln_beta_mult
                        )
                        print(
                            f"scaled BERT pre-LayerNorm by gamma={group_pre_ln_gamma_mult:g} "
                            f"beta={group_pre_ln_beta_mult:g} for {group_spec} "
                            f"({len(updated_pre_lns)} tensors)",
                            flush=True,
                        )
                    group_bias_mult = (
                        args.layermodule_linear_bias_mults[group_index]
                        if args.layermodule_linear_bias_mults is not None
                        else 1.0
                    )
                    if group_bias_mult != 1.0:
                        updated_biases = scale_bert_linear_bias_for_layer_module_group(
                            student, layers, aliases, group_bias_mult
                        )
                        print(
                            f"scaled BERT linear bias by {group_bias_mult:g} for {group_spec} "
                            f"({len(updated_biases)} tensors)",
                            flush=True,
                        )
                    group_ln_gamma_mult = (
                        args.layermodule_ln_gamma_mults[group_index]
                        if args.layermodule_ln_gamma_mults is not None
                        else 1.0
                    )
                    if group_ln_gamma_mult != 1.0:
                        updated_lns = scale_bert_layernorm_gamma_for_layer_module_group(
                            student, layers, aliases, group_ln_gamma_mult
                        )
                        print(
                            f"scaled BERT LayerNorm gamma by {group_ln_gamma_mult:g} for {group_spec} "
                            f"({len(updated_lns)} tensors)",
                            flush=True,
                        )
                    group_phase = f"distill_layermodule_cap_{stage_cap:g}_{group_spec}"
                    print(f"running distillation {group_phase} on {args.task}", flush=True)
                    recovery_run, selected_stage_state = train_distill_phase(
                        torch,
                        args,
                        student,
                        teacher,
                        train_loader,
                        eval_loader,
                        device,
                        phase=group_phase,
                        cap=stage_cap,
                        metric=metric,
                        target=target,
                        checkpoints=group_checkpoints,
                        learning_rate=group_learning_rate,
                        train_scope=args.recovery_train_scope,
                        ce_weight=group_ce_weight,
                        logit_mse_weight=group_logit_mse_weight,
                        margin_distill_weight=group_margin_weight,
                        hidden_distill_weight=group_hidden_weight,
                        attention_distill_weight=group_attention_weight,
                        rdrop_alpha=group_rdrop_alpha,
                        save_selection=args.grouped_stage_selection,
                        ramp_start_cap=previous_stage_cap if args.grouped_final_ramp else None,
                        ramp_end_cap=stage_cap if args.grouped_final_ramp else None,
                        ramp_steps=grouped_ramp_steps(group_checkpoints),
                        ramp_cap_setter=(
                            (lambda model, cap_value, layers=layers, aliases=aliases: set_learnable_cap_max_for_bert_layer_module_group(
                                model, cap_value, layers, aliases, init_fraction=None
                            ))
                            if args.grouped_final_ramp
                            else None
                        ),
                    )
                    recovery_runs.append(recovery_run)
                    if selected_stage_state is not None:
                        last_selected_state = selected_stage_state
                        load_student_state_with_caps(student, selected_stage_state)
                        student.to(device)
                print(f"running layer-module final distillation recovery cap {stage_cap:g} on {args.task}", flush=True)
                recovery_run, final_selected_state = train_distill_phase(
                    torch,
                    args,
                    student,
                    teacher,
                    train_loader,
                    eval_loader,
                    device,
                    phase=phase,
                    cap=stage_cap,
                    metric=metric,
                    target=target,
                    checkpoints=checkpoints,
                    learning_rate=stage_learning_rate,
                    train_scope=args.recovery_train_scope,
                    ce_weight=stage_ce_weight,
                    logit_mse_weight=stage_logit_mse_weight,
                    margin_distill_weight=stage_margin_weight,
                    hidden_distill_weight=stage_hidden_weight,
                    attention_distill_weight=stage_attention_weight,
                    rdrop_alpha=stage_rdrop_alpha,
                    save_selection=final_save_selection,
                )
                recovery_runs.append(recovery_run)
                if final_selected_state is not None:
                    last_selected_state = final_selected_state
                break
            if grouped_final_stage and module_groups:
                for group_index, (group_spec, aliases) in enumerate(zip(args.modulewise_final_groups, module_groups)):
                    if args.grouped_final_ramp:
                        updated = set_learnable_cap_max_for_bert_module_group(
                            student,
                            previous_stage_cap,
                            aliases,
                            init_fraction=None,
                        )
                        print(
                            f"ramping modulewise FN cap from {previous_stage_cap:g} to {stage_cap:g} "
                            f"on BERT modules {group_spec} ({updated} parametrizations)",
                            flush=True,
                        )
                    else:
                        updated = set_learnable_cap_max_for_bert_module_group(
                            student,
                            stage_cap,
                            aliases,
                            init_fraction=cap_lowering_init_fraction(args),
                        )
                        print(
                            f"lowered modulewise FN cap to {stage_cap:g} on BERT modules {group_spec} "
                            f"({updated} parametrizations)",
                            flush=True,
                        )
                    group_checkpoints = stage_checkpoints(args.modulewise_stage_steps[group_index], modulewise_checkpoint_interval)
                    group_learning_rate = (
                        args.modulewise_learning_rates[group_index]
                        if args.modulewise_learning_rates is not None
                        else args.recovery_learning_rate
                    )
                    group_ce_weight = (
                        args.modulewise_ce_weights[group_index]
                        if args.modulewise_ce_weights is not None
                        else stage_ce_weight
                    )
                    group_logit_mse_weight = (
                        args.modulewise_logit_mse_weights[group_index]
                        if args.modulewise_logit_mse_weights is not None
                        else stage_logit_mse_weight
                    )
                    group_margin_weight = (
                        args.modulewise_margin_distill_weights[group_index]
                        if args.modulewise_margin_distill_weights is not None
                        else stage_margin_weight
                    )
                    group_hidden_weight = (
                        args.modulewise_hidden_distill_weights[group_index]
                        if args.modulewise_hidden_distill_weights is not None
                        else stage_hidden_weight
                    )
                    group_attention_weight = (
                        args.modulewise_attention_distill_weights[group_index]
                        if args.modulewise_attention_distill_weights is not None
                        else stage_attention_weight
                    )
                    group_rdrop_alpha = (
                        args.modulewise_rdrop_alphas[group_index]
                        if args.modulewise_rdrop_alphas is not None
                        else stage_rdrop_alpha
                    )
                    group_phase = f"distill_modulewise_cap_{stage_cap:g}_modules_{group_spec}"
                    print(f"running distillation {group_phase} on {args.task}", flush=True)
                    recovery_run, selected_stage_state = train_distill_phase(
                        torch,
                        args,
                        student,
                        teacher,
                        train_loader,
                        eval_loader,
                        device,
                        phase=group_phase,
                        cap=stage_cap,
                        metric=metric,
                        target=target,
                        checkpoints=group_checkpoints,
                        learning_rate=group_learning_rate,
                        train_scope=args.recovery_train_scope,
                        ce_weight=group_ce_weight,
                        logit_mse_weight=group_logit_mse_weight,
                        margin_distill_weight=group_margin_weight,
                        hidden_distill_weight=group_hidden_weight,
                        attention_distill_weight=group_attention_weight,
                        rdrop_alpha=group_rdrop_alpha,
                        save_selection=args.grouped_stage_selection,
                        ramp_start_cap=previous_stage_cap if args.grouped_final_ramp else None,
                        ramp_end_cap=stage_cap if args.grouped_final_ramp else None,
                        ramp_steps=grouped_ramp_steps(group_checkpoints),
                        ramp_cap_setter=(
                            (lambda model, cap_value, aliases=aliases: set_learnable_cap_max_for_bert_module_group(
                                model, cap_value, aliases, init_fraction=None
                            ))
                            if args.grouped_final_ramp
                            else None
                        ),
                    )
                    recovery_runs.append(recovery_run)
                    if selected_stage_state is not None:
                        last_selected_state = selected_stage_state
                        load_student_state_with_caps(student, selected_stage_state)
                        student.to(device)
                print(f"running modulewise final distillation recovery cap {stage_cap:g} on {args.task}", flush=True)
                recovery_run, final_selected_state = train_distill_phase(
                    torch,
                    args,
                    student,
                    teacher,
                    train_loader,
                    eval_loader,
                    device,
                    phase=phase,
                    cap=stage_cap,
                    metric=metric,
                    target=target,
                    checkpoints=checkpoints,
                    learning_rate=stage_learning_rate,
                    train_scope=args.recovery_train_scope,
                    ce_weight=stage_ce_weight,
                    logit_mse_weight=stage_logit_mse_weight,
                    margin_distill_weight=stage_margin_weight,
                    hidden_distill_weight=stage_hidden_weight,
                    attention_distill_weight=stage_attention_weight,
                    rdrop_alpha=stage_rdrop_alpha,
                    save_selection=final_save_selection,
                )
                recovery_runs.append(recovery_run)
                if final_selected_state is not None:
                    last_selected_state = final_selected_state
                break
            if grouped_final_stage and layer_groups:
                for group_index, (group_spec, layers) in enumerate(zip(args.layerwise_final_groups, layer_groups)):
                    if args.grouped_final_ramp:
                        updated = set_learnable_cap_max_for_bert_layers(
                            student,
                            previous_stage_cap,
                            layers,
                            init_fraction=None,
                        )
                        print(
                            f"ramping grouped FN cap from {previous_stage_cap:g} to {stage_cap:g} "
                            f"on BERT layers {group_spec} ({updated} parametrizations)",
                            flush=True,
                        )
                    else:
                        updated = set_learnable_cap_max_for_bert_layers(
                            student,
                            stage_cap,
                            layers,
                            init_fraction=cap_lowering_init_fraction(args),
                        )
                        print(
                            f"lowered grouped FN cap to {stage_cap:g} on BERT layers {group_spec} "
                            f"({updated} parametrizations)",
                            flush=True,
                        )
                    group_checkpoints = stage_checkpoints(args.layerwise_stage_steps[group_index], layerwise_checkpoint_interval)
                    group_learning_rate = (
                        args.layerwise_learning_rates[group_index]
                        if args.layerwise_learning_rates is not None
                        else args.recovery_learning_rate
                    )
                    group_ce_weight = (
                        args.layerwise_ce_weights[group_index]
                        if args.layerwise_ce_weights is not None
                        else stage_ce_weight
                    )
                    group_phase = f"distill_layerwise_cap_{stage_cap:g}_layers_{group_spec}"
                    print(f"running distillation {group_phase} on {args.task}", flush=True)
                    recovery_run, selected_stage_state = train_distill_phase(
                        torch,
                        args,
                        student,
                        teacher,
                        train_loader,
                        eval_loader,
                        device,
                        phase=group_phase,
                        cap=stage_cap,
                        metric=metric,
                        target=target,
                        checkpoints=group_checkpoints,
                        learning_rate=group_learning_rate,
                        train_scope=args.recovery_train_scope,
                        ce_weight=group_ce_weight,
                        logit_mse_weight=stage_logit_mse_weight,
                        margin_distill_weight=stage_margin_weight,
                        hidden_distill_weight=stage_hidden_weight,
                        attention_distill_weight=stage_attention_weight,
                        rdrop_alpha=stage_rdrop_alpha,
                        save_selection=args.grouped_stage_selection,
                        ramp_start_cap=previous_stage_cap if args.grouped_final_ramp else None,
                        ramp_end_cap=stage_cap if args.grouped_final_ramp else None,
                        ramp_steps=grouped_ramp_steps(group_checkpoints),
                        ramp_cap_setter=(
                            (lambda model, cap_value, layers=layers: set_learnable_cap_max_for_bert_layers(
                                model, cap_value, layers, init_fraction=None
                            ))
                            if args.grouped_final_ramp
                            else None
                        ),
                    )
                    recovery_runs.append(recovery_run)
                    if selected_stage_state is not None:
                        last_selected_state = selected_stage_state
                        load_student_state_with_caps(student, selected_stage_state)
                        student.to(device)
                print(f"running grouped final distillation recovery cap {stage_cap:g} on {args.task}", flush=True)
                recovery_run, final_selected_state = train_distill_phase(
                    torch,
                    args,
                    student,
                    teacher,
                    train_loader,
                    eval_loader,
                    device,
                    phase=phase,
                    cap=stage_cap,
                    metric=metric,
                    target=target,
                    checkpoints=checkpoints,
                    learning_rate=stage_learning_rate,
                    train_scope=args.recovery_train_scope,
                    ce_weight=stage_ce_weight,
                    logit_mse_weight=stage_logit_mse_weight,
                    margin_distill_weight=stage_margin_weight,
                    hidden_distill_weight=stage_hidden_weight,
                    attention_distill_weight=stage_attention_weight,
                    rdrop_alpha=stage_rdrop_alpha,
                    save_selection=final_save_selection,
                )
                recovery_runs.append(recovery_run)
                if final_selected_state is not None:
                    last_selected_state = final_selected_state
                break
            stage_train_scope = (
                args.anneal_stage_train_scopes[stage_index]
                if (not final_stage and args.anneal_stage_train_scopes is not None)
                else args.recovery_train_scope
            )
            stage_ramp_start_cap = stage_cap if (args.anneal_stage_ramp_to_next_cap and not final_stage) else None
            stage_ramp_end_cap = schedule[stage_index + 1] if (args.anneal_stage_ramp_to_next_cap and not final_stage) else None
            stage_ramp_steps = max(checkpoints) if stage_ramp_start_cap is not None and checkpoints else None
            ramp_note = f" ramp_to={stage_ramp_end_cap:g}" if stage_ramp_end_cap is not None else ""
            print(f"running distillation {phase} cap {stage_cap:g}{ramp_note} on {args.task}", flush=True)
            recovery_run, selected_stage_state = train_distill_phase(
                torch,
                args,
                student,
                teacher,
                train_loader,
                eval_loader,
                device,
                phase=phase,
                cap=stage_cap,
                metric=metric,
                target=target,
                checkpoints=checkpoints,
                learning_rate=stage_learning_rate,
                train_scope=stage_train_scope,
                ce_weight=stage_ce_weight,
                logit_mse_weight=stage_logit_mse_weight,
                margin_distill_weight=stage_margin_weight,
                hidden_distill_weight=stage_hidden_weight,
                attention_distill_weight=stage_attention_weight,
                rdrop_alpha=stage_rdrop_alpha,
                save_selection=final_save_selection if final_stage else args.anneal_selection,
                lookahead_cap=(schedule[stage_index + 1] if (not final_stage and args.anneal_selection == "lookahead") else None),
                ramp_start_cap=stage_ramp_start_cap,
                ramp_end_cap=stage_ramp_end_cap,
                ramp_steps=stage_ramp_steps,
            )
            recovery_runs.append(recovery_run)
            if selected_stage_state is not None:
                last_selected_state = selected_stage_state
            if final_stage:
                final_selected_state = selected_stage_state
            elif selected_stage_state is not None:
                load_student_state_with_caps(student, selected_stage_state)
                student.to(device)
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted; writing partial recovery artifact", flush=True)
    if not recovery_runs:
        if interrupted:
            print("interrupted before any recovery phase completed; no partial artifact written", flush=True)
            return
        raise RuntimeError("no recovery runs recorded")
    recovery_run = recovery_runs[-1]
    output = ensure_parent(args.output)
    payload = {
        **now_metadata(),
        "model": args.model,
        "task": args.task,
        "dataset": "glue",
        "interrupted": interrupted,
        "seed": args.seed,
        "device": str(device),
        "cache_dir": args.cache_dir,
        "offline": args.offline,
        "teacher_state_in": args.teacher_state_in,
        "save_teacher_state": args.save_teacher_state,
        "student_state_in": args.student_state_in,
        "save_student_state": args.save_student_state,
        "final_save_selection": args.final_save_selection,
        "cap": args.cap,
        "anneal_caps": schedule,
        "anneal_stage_steps": args.anneal_stage_steps,
        "anneal_stage_learning_rates": args.anneal_stage_learning_rates,
        "anneal_stage_ramp_to_next_cap": args.anneal_stage_ramp_to_next_cap,
        "anneal_stage_ramp_preserve_lower_caps": args.anneal_stage_ramp_preserve_lower_caps,
        "stage_lower_preserve_lower_caps": args.stage_lower_preserve_lower_caps,
        "anneal_stage_train_scopes": args.anneal_stage_train_scopes,
        "anneal_checkpoint_interval": args.anneal_checkpoint_interval,
        "metric": metric,
        "baseline_target": target,
        "baseline_selection": args.baseline_selection,
        "warm_start_selection": args.warm_start_selection,
        "anneal_selection": args.anneal_selection,
        "tolerance": args.tolerance,
        "threshold": threshold_value(target, metric, args.tolerance),
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "baseline_learning_rate": args.baseline_learning_rate,
        "recovery_learning_rate": args.recovery_learning_rate,
        "recovery_scheduler_steps": args.recovery_scheduler_steps,
        "weight_decay": args.weight_decay,
        "classifier_lr_mult": args.classifier_lr_mult,
        "norm_lr_mult": args.norm_lr_mult,
        "cap_scalar_lr_mult": args.cap_scalar_lr_mult,
        "param_lr_mult_patterns": args.param_lr_mult_patterns,
        "param_lr_mults": args.param_lr_mults,
        "optimizer": args.optimizer,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_eps": args.adam_eps,
        "baseline_train_scope": args.baseline_train_scope,
        "recovery_train_scope": args.recovery_train_scope,
        "extra_trainable_patterns": args.extra_trainable_patterns,
        "dtype": args.dtype,
        "cap_norm_dtype": args.cap_norm_dtype,
        "cap_init_fraction": args.cap_init_fraction,
        "preserve_cap_fraction_on_lower": args.preserve_cap_fraction_on_lower,
        "freeze_cap_scalars": args.freeze_cap_scalars,
        "student_dropout_p": args.student_dropout_p,
        "distill_temperature": args.distill_temperature,
        "ce_weight": args.ce_weight,
        "ce_label_source": args.ce_label_source,
        "label_smoothing": args.label_smoothing,
        "focal_gamma": args.focal_gamma,
        "teacher_disagree_ce_multiplier": args.teacher_disagree_ce_multiplier,
        "class_weights": args.class_weights,
        "label_margin_floor_weight": args.label_margin_floor_weight,
        "label_margin_floor": args.label_margin_floor,
        "logit_mse_weight": args.logit_mse_weight,
        "anneal_stage_logit_mse_weights": args.anneal_stage_logit_mse_weights,
        "recovery_logit_mse_weight": args.recovery_logit_mse_weight,
        "layermodule_logit_mse_weights": args.layermodule_logit_mse_weights,
        "modulewise_logit_mse_weights": args.modulewise_logit_mse_weights,
        "margin_distill_weight": args.margin_distill_weight,
        "teacher_margin_floor_weight": args.teacher_margin_floor_weight,
        "teacher_margin_floor": args.teacher_margin_floor,
        "anneal_stage_margin_distill_weights": args.anneal_stage_margin_distill_weights,
        "recovery_margin_distill_weight": args.recovery_margin_distill_weight,
        "layermodule_margin_distill_weights": args.layermodule_margin_distill_weights,
        "modulewise_margin_distill_weights": args.modulewise_margin_distill_weights,
        "hidden_distill_weight": args.hidden_distill_weight,
        "anneal_stage_hidden_distill_weights": args.anneal_stage_hidden_distill_weights,
        "recovery_hidden_distill_weight": args.recovery_hidden_distill_weight,
        "layermodule_hidden_distill_weights": args.layermodule_hidden_distill_weights,
        "hidden_distill_layers": args.hidden_distill_layers,
        "hidden_distill_mode": args.hidden_distill_mode,
        "hidden_distill_normalize": args.hidden_distill_normalize,
        "attention_distill_weight": args.attention_distill_weight,
        "attention_distill_layers": args.attention_distill_layers,
        "anneal_stage_attention_distill_weights": args.anneal_stage_attention_distill_weights,
        "recovery_attention_distill_weight": args.recovery_attention_distill_weight,
        "layermodule_attention_distill_weights": args.layermodule_attention_distill_weights,
        "rdrop_alpha": args.rdrop_alpha,
        "ema_decay": args.ema_decay,
        "ema_start_step": args.ema_start_step,
        "sam_rho": args.sam_rho,
        "sam_adaptive": args.sam_adaptive,
        "sam_eps": args.sam_eps,
        "swa_start_step": args.swa_start_step,
        "swa_every": args.swa_every,
        "anneal_stage_rdrop_alphas": args.anneal_stage_rdrop_alphas,
        "recovery_rdrop_alpha": args.recovery_rdrop_alpha,
        "layermodule_rdrop_alphas": args.layermodule_rdrop_alphas,
        "layermodule_ln_gamma_mults": args.layermodule_ln_gamma_mults,
        "layermodule_pre_ln_gamma_mults": args.layermodule_pre_ln_gamma_mults,
        "layermodule_pre_ln_beta_mults": args.layermodule_pre_ln_beta_mults,
        "layermodule_linear_bias_mults": args.layermodule_linear_bias_mults,
        "anneal_stage_ce_weights": args.anneal_stage_ce_weights,
        "recovery_ce_weight": args.recovery_ce_weight,
        "layerwise_final_groups": args.layerwise_final_groups,
        "layerwise_stage_steps": args.layerwise_stage_steps,
        "layerwise_learning_rates": args.layerwise_learning_rates,
        "layerwise_ce_weights": args.layerwise_ce_weights,
        "layerwise_checkpoint_interval": args.layerwise_checkpoint_interval,
        "modulewise_final_groups": args.modulewise_final_groups,
        "modulewise_stage_steps": args.modulewise_stage_steps,
        "modulewise_learning_rates": args.modulewise_learning_rates,
        "modulewise_ce_weights": args.modulewise_ce_weights,
        "modulewise_hidden_distill_weights": args.modulewise_hidden_distill_weights,
        "modulewise_attention_distill_weights": args.modulewise_attention_distill_weights,
        "modulewise_rdrop_alphas": args.modulewise_rdrop_alphas,
        "modulewise_checkpoint_interval": args.modulewise_checkpoint_interval,
        "layermodule_final_groups": args.layermodule_final_groups,
        "layermodule_stage_steps": args.layermodule_stage_steps,
        "layermodule_learning_rates": args.layermodule_learning_rates,
        "layermodule_ce_weights": args.layermodule_ce_weights,
        "layermodule_checkpoint_interval": args.layermodule_checkpoint_interval,
        "grouped_final_ramp": args.grouped_final_ramp,
        "grouped_ramp_fraction": args.grouped_ramp_fraction,
        "grouped_stage_selection": args.grouped_stage_selection,
        "gradient_checkpointing": args.gradient_checkpointing,
        "device_map": args.device_map,
        "force_decoder_wrapper": args.force_decoder_wrapper,
        "load_in_4bit": args.load_in_4bit,
        "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
        "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype or args.dtype,
        "max_length": args.max_length,
        "pad_to_max_length": args.pad_to_max_length,
        "tensor_sequence_length": args.max_length if args.pad_to_max_length else None,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "applied_caps": applied_caps,
        "weight_stats_before": weight_stats_before,
        "weight_stats_after_cap": weight_stats_after_cap,
        "package_versions": package_versions(),
        "baseline_run": baseline_run,
        "recovery_runs": recovery_runs,
        "recovery_run": recovery_run,
        "minimal_recovery": summarize_recovery(recovery_run),
    }
    if args.save_student_state:
        state_output = ensure_parent(args.save_student_state)
        if final_selected_state is not None:
            state_to_save = final_selected_state
        elif interrupted and last_selected_state is not None:
            state_to_save = last_selected_state
            payload["saved_student_state_source"] = "last_completed_selected_checkpoint"
        else:
            state_to_save = clone_student_state_with_caps(student)
        if isinstance(state_to_save, dict) and "state_dict" in state_to_save:
            wrapped_state_to_save = state_to_save
        else:
            cap_stats_to_save = collect_learnable_cap_stats(student)
            wrapped_state_to_save = {
                "state_dict": state_to_save,
                "cap_max_gamma_by_name": {row["name"]: row["max_gamma"] for row in cap_stats_to_save},
                "cap_stats": cap_stats_to_save,
            }
        torch.save(wrapped_state_to_save, state_output)
        payload["saved_student_state"] = str(state_output)
        print(f"saved student state {state_output}", flush=True)
    write_json(output, payload)
    write_json(metadata_path(output), {k: v for k, v in payload.items() if k not in {"baseline_run", "recovery_run", "recovery_runs"}})
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
