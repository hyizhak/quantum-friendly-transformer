#!/usr/bin/env python
"""Aggressive GLUE recovery sweeps for 7B decoder sequence classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path
import gc
import math
import time

from quantum_friendly_transformer.norm_transformer import (
    ATTENTION_SCALING_CHOICES,
    DEFAULT_MAX_ATTENTION_SCALE,
    DEFAULT_MIN_ATTENTION_SCALE,
    apply_attention_scaling,
    apply_learnable_frobenius_caps,
    attention_scale_native_regularization,
    collect_attention_scaling_stats,
    collect_learnable_cap_stats,
    project_attention_scales_,
    set_attention_scales_to_native_,
    target_weight_modules,
)

from quantum_friendly_transformer.experiments.lib.common import (
    GLUE_TEXT_COLUMNS,
    compute_classification_metrics,
    ensure_parent,
    glue_num_labels,
    glue_split_name,
    metadata_path,
    now_metadata,
    package_versions,
    seed_everything,
    write_json,
)


HIGHER_IS_BETTER = {
    "accuracy": True,
    "f1": True,
    "matthews": True,
    "pearson": True,
    "spearman": True,
    "eval_loss": False,
}

TRAIN_SCOPE_CHOICES = [
    "classifier_bias",
    "classifier_bias_caps",
    "classifier",
    "classifier_caps",
    "head_caps",
    "head_pooler_caps",
    "target_caps",
    "all",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--task", required=True, choices=sorted(GLUE_TEXT_COLUMNS))
    parser.add_argument("--caps", type=float, nargs="+", default=None)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--metric", default=None)
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--baseline-target", type=float, default=None)
    parser.add_argument("--baseline-selection", choices=["final", "best"], default="final")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--attention-scaling",
        choices=ATTENTION_SCALING_CHOICES,
        default="native",
        help=(
            "Attention-score denominator. 'native' preserves the backbone, "
            "'sqrt_head_dim' uses sqrt(d_head), 'sequence_length' uses the fixed tensor length N, "
            "and 'alpha_s_alpha_w' uses alpha_s^2 * alpha_w^2."
        ),
    )
    parser.add_argument(
        "--alpha-s",
        type=float,
        default=None,
        help="Positive activation-norm bound alpha_s for --attention-scaling alpha_s_alpha_w.",
    )
    parser.add_argument(
        "--alpha-w",
        type=float,
        default=None,
        help="Positive weight-norm bound alpha_w for --attention-scaling alpha_s_alpha_w.",
    )
    parser.add_argument(
        "--trainable-attention-scaling",
        action="store_true",
        help=(
            "Learn one positive score multiplier per attention layer, initialized "
            "from alpha_s^2 * alpha_w^2. Supported only for alpha_s_alpha_w mode."
        ),
    )
    parser.add_argument(
        "--shared-attention-scale",
        action="store_true",
        help="Use one learned alpha score multiplier shared by every attention layer.",
    )
    parser.add_argument(
        "--enforce-alpha-denominator-bound",
        action="store_true",
        help=(
            "Enforce D >= alpha_s^2 * alpha_w^2 throughout training, equivalently "
            "cap the learned score multiplier at 1 / (alpha_s^2 * alpha_w^2). "
            "This forbids relaxation toward a smaller/native denominator."
        ),
    )
    parser.add_argument(
        "--attention-scale-learning-rate",
        type=float,
        default=1e-2,
        help="Learning rate for trainable attention scales (always zero weight decay).",
    )
    parser.add_argument(
        "--attention-scale-adam-eps",
        type=float,
        default=1e-12,
        help="AdamW epsilon for the tiny-gradient attention-scale parameter group.",
    )
    parser.add_argument(
        "--attention-scale-native-regularization",
        type=float,
        default=0.0,
        help=(
            "Coefficient for squared log-distance between learned and native "
            "attention score scales. Zero disables the soft prior."
        ),
    )
    parser.add_argument(
        "--attention-scale-prior-warmup-steps",
        type=int,
        default=0,
        help=(
            "Controller-only steps minimizing the native-scale prior before "
            "task fine-tuning. Base-model weights are untouched."
        ),
    )
    parser.add_argument(
        "--attention-scale-prior-warmup-learning-rate",
        type=float,
        default=0.3,
        help="AdamW learning rate for controller-only native-prior warmup.",
    )
    parser.add_argument(
        "--attention-scale-prior-warmup-snap-native",
        action="store_true",
        help="After prior warmup, set the controller exactly to its native optimum.",
    )
    parser.add_argument(
        "--freeze-attention-scale-after-prior-warmup",
        action="store_true",
        help="Freeze the learned controller for downstream task fine-tuning.",
    )
    parser.add_argument(
        "--attention-scale-min",
        type=float,
        default=DEFAULT_MIN_ATTENTION_SCALE,
        help="Strict lower bound for each learned attention-score multiplier.",
    )
    parser.add_argument(
        "--attention-scale-max",
        type=float,
        default=DEFAULT_MAX_ATTENTION_SCALE,
        help="Strict upper bound for each learned attention-score multiplier.",
    )
    parser.add_argument(
        "--alpha-s-calibration-batches",
        type=int,
        default=0,
        help=(
            "Number of SST-2 evaluation batches used to measure exact pre-attention "
            "token L2 maxima before installing alpha_s_alpha_w scaling."
        ),
    )
    parser.add_argument(
        "--alpha-s-calibration-margin",
        type=float,
        default=1.05,
        help="Safety multiplier applied to the observed attention-input maximum.",
    )
    parser.add_argument(
        "--pad-to-max-length",
        action="store_true",
        help="Pad every batch to --max-length so the tensor sequence dimension N is fixed.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--lr-scheduler",
        choices=["linear", "constant"],
        default="linear",
        help="Task-training learning-rate schedule.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--train-scope", choices=TRAIN_SCOPE_CHOICES, default="target_caps")
    parser.add_argument("--optimizer", choices=["sgd", "adamw", "adafactor"], default="sgd")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--cap-norm-dtype", choices=["native", "float32"], default="native")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--force-decoder-wrapper", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb-4bit-compute-dtype", choices=["bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--peft-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--early-stop-on-recovery",
        action="store_true",
        help="Stop after the first evaluated checkpoint that meets the recovery threshold.",
    )
    parser.add_argument(
        "--save-recovery-checkpoint",
        action="store_true",
        help="Save trainable tensors when a variant first meets its recovery threshold.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def validate_attention_scaling_args(args: argparse.Namespace) -> None:
    """Validate mode-specific scaling arguments before loading data or a model."""

    uses_alpha_bounds = args.attention_scaling == "alpha_s_alpha_w"
    supplied_alpha_bound = args.alpha_s is not None or args.alpha_w is not None
    enforce_alpha_bound = getattr(args, "enforce_alpha_denominator_bound", False)
    if args.trainable_attention_scaling and not uses_alpha_bounds:
        raise ValueError(
            "--trainable-attention-scaling requires "
            "--attention-scaling alpha_s_alpha_w"
        )
    if args.shared_attention_scale and not args.trainable_attention_scaling:
        raise ValueError(
            "--shared-attention-scale requires --trainable-attention-scaling"
        )
    if enforce_alpha_bound and (
        not args.trainable_attention_scaling or not uses_alpha_bounds
    ):
        raise ValueError(
            "--enforce-alpha-denominator-bound requires trainable "
            "alpha_s_alpha_w scaling"
        )
    if enforce_alpha_bound and args.attention_scale_prior_warmup_steps:
        raise ValueError(
            "--enforce-alpha-denominator-bound is incompatible with native-prior "
            "warmup because the native scale would violate the alpha denominator"
        )
    if enforce_alpha_bound and args.attention_scale_prior_warmup_snap_native:
        raise ValueError(
            "--enforce-alpha-denominator-bound is incompatible with snapping the "
            "attention scale to native"
        )
    if enforce_alpha_bound and args.attention_scale_native_regularization > 0:
        raise ValueError(
            "--enforce-alpha-denominator-bound is incompatible with native-scale "
            "regularization"
        )
    if (
        args.attention_scale_prior_warmup_steps
        and not args.trainable_attention_scaling
    ):
        raise ValueError(
            "--attention-scale-prior-warmup-steps requires "
            "--trainable-attention-scaling"
        )
    if (
        args.freeze_attention_scale_after_prior_warmup
        and not args.attention_scale_prior_warmup_steps
    ):
        raise ValueError(
            "--freeze-attention-scale-after-prior-warmup requires positive "
            "--attention-scale-prior-warmup-steps"
        )
    if args.trainable_attention_scaling and args.device_map is not None:
        raise ValueError(
            "trainable attention scaling does not support --device-map dispatch; "
            "load the model on one device so attached controllers cannot be offloaded"
        )
    if args.alpha_s_calibration_batches and not uses_alpha_bounds:
        raise ValueError(
            "--alpha-s-calibration-batches is only valid with "
            "--attention-scaling alpha_s_alpha_w"
        )
    if not uses_alpha_bounds:
        if supplied_alpha_bound:
            raise ValueError(
                "--alpha-s and --alpha-w are only valid with "
                "--attention-scaling alpha_s_alpha_w"
            )
        return

    if args.alpha_s is None or args.alpha_w is None:
        raise ValueError(
            "--attention-scaling alpha_s_alpha_w requires both --alpha-s and --alpha-w"
        )
    for option, value in (("--alpha-s", args.alpha_s), ("--alpha-w", args.alpha_w)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{option} must be a finite positive number")
    denominator = args.alpha_s * args.alpha_s * args.alpha_w * args.alpha_w
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("alpha_s^2 * alpha_w^2 must be a finite positive number")
    if args.alpha_s_calibration_batches < 0:
        raise ValueError("--alpha-s-calibration-batches cannot be negative")
    if (
        not math.isfinite(args.alpha_s_calibration_margin)
        or args.alpha_s_calibration_margin < 1.0
    ):
        raise ValueError("--alpha-s-calibration-margin must be finite and at least 1")
    if getattr(args, "trainable_attention_scaling", False):
        for option, value in (
            ("--attention-scale-learning-rate", args.attention_scale_learning_rate),
            ("--attention-scale-adam-eps", args.attention_scale_adam_eps),
            ("--attention-scale-min", args.attention_scale_min),
            ("--attention-scale-max", args.attention_scale_max),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{option} must be a finite positive number")
        if (
            not math.isfinite(args.attention_scale_native_regularization)
            or args.attention_scale_native_regularization < 0
        ):
            raise ValueError(
                "--attention-scale-native-regularization must be finite and nonnegative"
            )
        if args.attention_scale_prior_warmup_steps < 0:
            raise ValueError("--attention-scale-prior-warmup-steps cannot be negative")
        if args.attention_scale_prior_warmup_steps:
            if args.attention_scale_native_regularization <= 0:
                raise ValueError(
                    "--attention-scale-prior-warmup-steps requires positive "
                    "--attention-scale-native-regularization"
                )
            if (
                not math.isfinite(args.attention_scale_prior_warmup_learning_rate)
                or args.attention_scale_prior_warmup_learning_rate <= 0
            ):
                raise ValueError(
                    "--attention-scale-prior-warmup-learning-rate must be finite and positive"
                )
        if args.attention_scale_max <= args.attention_scale_min:
            raise ValueError("--attention-scale-max must exceed --attention-scale-min")
        initial_scale = 1.0 / denominator
        if not args.attention_scale_min < initial_scale < args.attention_scale_max:
            raise ValueError(
                "the alpha-initialized score scale must lie strictly between "
                "--attention-scale-min and --attention-scale-max"
            )


def validate_optimizer_args(args: argparse.Namespace) -> None:
    for option, value in (
        ("--adam-beta1", args.adam_beta1),
        ("--adam-beta2", args.adam_beta2),
    ):
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{option} must be finite and in [0, 1)")
    if not math.isfinite(args.adam_eps) or args.adam_eps <= 0:
        raise ValueError("--adam-eps must be finite and positive")


def attention_scale_denominator(args: argparse.Namespace):
    """Return the configured score denominator in a JSON-serializable form."""

    if args.attention_scaling == "sequence_length":
        return args.max_length
    if args.attention_scaling == "alpha_s_alpha_w":
        return args.alpha_s * args.alpha_s * args.alpha_w * args.alpha_w
    return args.attention_scaling


def primary_metric(task: str) -> str:
    if task in {"mrpc", "qqp"}:
        return "f1"
    if task == "cola":
        return "matthews"
    if task == "stsb":
        return "pearson"
    return "accuracy"


def threshold_value(target: float | None, metric: str, tolerance: float) -> float | None:
    if target is None:
        return None
    if HIGHER_IS_BETTER.get(metric, True):
        return target - tolerance
    return target + tolerance


def meets_threshold(value: float | None, target: float | None, metric: str, tolerance: float) -> bool | None:
    if target is None or value is None:
        return None
    threshold = threshold_value(target, metric, tolerance)
    if HIGHER_IS_BETTER.get(metric, True):
        return value >= threshold
    return value <= threshold


def baseline_target_from_json(path: str | None, metric: str) -> float | None:
    if not path:
        return None
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text())
    if payload.get("baseline_target") is not None:
        return float(payload["baseline_target"])
    baseline = payload.get("baseline_run")
    if baseline:
        checkpoints = baseline.get("checkpoints", [])
        if checkpoints:
            return float(checkpoints[-1]["metric_value"])
    for row in payload.get("results", []):
        if row.get("kind") == "baseline" and row.get("checkpoints"):
            return float(row["checkpoints"][-1]["metric_value"])
    raise ValueError(f"could not find baseline target in {path}")


def select_baseline_target(checkpoints: list[dict], metric: str, selection: str) -> float:
    if not checkpoints:
        raise ValueError("baseline run has no checkpoints")
    if selection == "final":
        return float(checkpoints[-1]["metric_value"])
    key = (lambda row: row["metric_value"]) if HIGHER_IS_BETTER.get(metric, True) else (lambda row: -row["metric_value"])
    return float(max(checkpoints, key=key)["metric_value"])


def pick_device(torch, requested: str):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def torch_dtype(torch, dtype: str):
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    return torch.float32


def prepare_dataset(args, tokenizer, load_dataset):
    train = load_dataset("glue", args.task, split="train", cache_dir=args.cache_dir)
    eval_split = glue_split_name(args.task, "validation")
    eval_ds = load_dataset("glue", args.task, split=eval_split, cache_dir=args.cache_dir)
    if args.max_train_samples is not None:
        train = train.select(range(min(args.max_train_samples, len(train))))
    if args.max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    col1, col2 = GLUE_TEXT_COLUMNS[args.task]

    def tokenize(batch):
        kwargs = {"truncation": True, "max_length": args.max_length}
        if col2 is None:
            encoded = tokenizer(batch[col1], **kwargs)
        else:
            encoded = tokenizer(batch[col1], batch[col2], **kwargs)
        encoded["labels"] = batch["label"]
        return encoded

    keep = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    train = train.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)
    train = train.remove_columns([col for col in train.column_names if col not in keep])
    eval_ds = eval_ds.remove_columns([col for col in eval_ds.column_names if col not in keep])
    return train, eval_ds


def make_loaders(torch, args, tokenizer, train_ds, eval_ds):
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    if args.pad_to_max_length:
        collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=args.max_length)
    else:
        collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, collate_fn=collator)
    eval_loader = DataLoader(eval_ds, shuffle=False, batch_size=args.eval_batch_size, collate_fn=collator)
    return train_loader, eval_loader


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def model_forward_kwargs(model, batch):
    import inspect

    parameters = inspect.signature(model.forward).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return batch
    allowed = set(parameters)
    return {key: value for key, value in batch.items() if key in allowed}


def call_model(model, batch):
    return model(**model_forward_kwargs(model, batch))


def fast_weight_stats(model) -> list[dict]:
    rows = []
    for name, module in target_weight_modules(model):
        weight = module.weight.detach()
        rows.append(
            {
                "name": f"{name}.weight",
                "frobenius_norm": float(weight.float().norm().cpu().item()),
                "shape": list(weight.shape),
            }
        )
    return rows


def cap_summary(cap_stats: list[dict]) -> dict:
    if not cap_stats:
        return {}
    gammas = [row["gamma"] for row in cap_stats]
    fracs = [row["gamma_fraction"] for row in cap_stats]
    return {
        "gamma_min": min(gammas),
        "gamma_mean": sum(gammas) / len(gammas),
        "gamma_max": max(gammas),
        "gamma_fraction_min": min(fracs),
        "gamma_fraction_mean": sum(fracs) / len(fracs),
        "gamma_fraction_max": max(fracs),
    }


def attention_scale_summary(scale_stats: list[dict]) -> dict:
    if not scale_stats:
        return {}
    scales = [row["score_scale"] for row in scale_stats]
    denominators = [row["score_denominator"] for row in scale_stats]
    return {
        "score_scale_min": min(scales),
        "score_scale_mean": sum(scales) / len(scales),
        "score_scale_max": max(scales),
        "score_denominator_min": min(denominators),
        "score_denominator_mean": sum(denominators) / len(denominators),
        "score_denominator_max": max(denominators),
    }


def evaluate(torch, model, loader, device, task: str) -> dict:
    model.eval()
    losses = []
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = call_model(model, batch)
            losses.append(float(outputs.loss.detach().cpu().item()))
            logits = outputs.logits.detach().cpu()
            label = batch["labels"].detach().cpu()
            if task == "stsb":
                pred = logits.squeeze(-1).numpy().tolist()
            else:
                pred = logits.argmax(dim=-1).numpy().tolist()
            preds.extend(pred)
            labels.extend(label.numpy().tolist())
    metrics = compute_classification_metrics(task, preds, labels)
    metrics["eval_loss"] = sum(losses) / max(1, len(losses))
    return metrics


def calibrate_attention_input_bound(torch, model, loader, device, max_batches: int) -> dict:
    """Measure exact tensors received by supported attention modules on SST-2.

    Creating a ``DataLoader`` iterator advances PyTorch's RNG even when the
    loader is not shuffled.  Preserve both CPU and CUDA RNG state so this
    diagnostic pass cannot change the subsequent task-training sample order
    (or any stochastic model operation).  This matters when comparing an
    alpha-scaled run with an otherwise identical native-attention recovery.
    """

    supported_types = {
        "GPTNeoSelfAttention",
        "OlmoAttention",
        "Olmo2Attention",
        "MistralAttention",
    }
    maxima = {}
    calls = {}
    handles = []

    def make_hook(name):
        def hook(_module, inputs, kwargs):
            hidden = kwargs.get("hidden_states")
            if hidden is None:
                if not inputs:
                    raise RuntimeError(f"attention layer {name} received no hidden states")
                hidden = inputs[0]
            value = float(
                torch.linalg.vector_norm(hidden.detach().float(), dim=-1)
                .amax()
                .cpu()
                .item()
            )
            maxima[name] = max(maxima.get(name, 0.0), value)
            calls[name] = calls.get(name, 0) + 1

        return hook

    for name, module in model.named_modules():
        if type(module).__name__ in supported_types:
            handles.append(
                module.register_forward_pre_hook(make_hook(name), with_kwargs=True)
            )
    if not handles:
        raise ValueError("no supported attention layers found for alpha_s calibration")

    was_training = model.training
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    model.eval()
    batches_seen = 0
    try:
        with torch.no_grad():
            for batch in loader:
                inputs = move_batch(batch, device)
                inputs.pop("labels", None)
                call_model(model, inputs)
                batches_seen += 1
                if batches_seen >= max_batches:
                    break
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    if not maxima or any(count != batches_seen for count in calls.values()):
        raise RuntimeError(
            "alpha_s calibration did not observe every attention layer on every batch"
        )
    rows = [
        {"name": name, "max_token_l2": maxima[name], "calls": calls[name]}
        for name in sorted(maxima)
    ]
    return {
        "batches": batches_seen,
        "layers": rows,
        "observed_max_token_l2": max(maxima.values()),
    }


def _is_classifier_parameter_name(name: str) -> bool:
    """Return whether a parameter belongs to a score/classifier head."""

    lowered = name.lower()
    return (
        lowered.startswith("score.")
        or ".score." in lowered
        or "classifier" in lowered
    )


def configure_trainable(
    model,
    train_scope: str,
    capped: bool,
    peft_lora: bool = False,
    trainable_attention_scaling: bool = False,
) -> int:
    if peft_lora:
        for name, param in model.named_parameters():
            lowered = name.lower()
            param.requires_grad = "lora_" in lowered or "modules_to_save" in lowered
    else:
        for param in model.parameters():
            param.requires_grad = train_scope == "all"

    if train_scope != "all":
        for name, param in model.named_parameters():
            lowered = name.lower()
            is_classifier = _is_classifier_parameter_name(name)
            is_classifier_bias = is_classifier and lowered.endswith(".bias")
            is_norm = param.ndim == 1 and ("norm" in lowered or "ln_" in lowered)
            is_cap_scalar = (
                "parametrizations.weight." in lowered and lowered.endswith(".g")
            )
            is_target_weight = "parametrizations.weight.original" in lowered
            if train_scope in {"classifier", "classifier_caps"} and is_classifier:
                param.requires_grad = True
            elif train_scope in {"classifier_bias", "classifier_bias_caps"} and is_classifier_bias:
                param.requires_grad = True
            elif train_scope in {"classifier_caps", "classifier_bias_caps"} and is_cap_scalar:
                param.requires_grad = True
            elif train_scope in {"head_caps", "head_pooler_caps", "target_caps"} and is_classifier:
                param.requires_grad = True
            elif train_scope == "head_pooler_caps" and "pooler" in lowered:
                param.requires_grad = True
            elif train_scope in {"head_caps", "head_pooler_caps", "target_caps"} and is_norm:
                param.requires_grad = True
            elif train_scope in {"head_caps", "head_pooler_caps", "target_caps"} and is_cap_scalar:
                param.requires_grad = True
            elif train_scope == "target_caps" and is_target_weight:
                param.requires_grad = True

    if train_scope == "target_caps" and not capped and not peft_lora:
        target_param_ids = {id(module.weight) for _, module in target_weight_modules(model)}
        for param in model.parameters():
            if id(param) in target_param_ids:
                param.requires_grad = True

    if trainable_attention_scaling:
        for name, param in model.named_parameters():
            if "_qft_learnable_attention_scale." in name:
                param.requires_grad = True

    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def make_optimizer(
    args,
    trainable_params,
    attention_scale_params=None,
):
    trainable_params = list(trainable_params)
    attention_scale_params = list(attention_scale_params or [])
    parameter_lists = {
        "trainable": trainable_params,
        "attention-scale": attention_scale_params,
    }
    for label, params in parameter_lists.items():
        if len({id(param) for param in params}) != len(params):
            raise ValueError(f"duplicate parameters in {label} optimizer input")

    trainable_ids = {id(param) for param in trainable_params}
    scale_ids = {id(param) for param in attention_scale_params}
    if not scale_ids.issubset(trainable_ids):
        raise ValueError(
            "attention-scale optimizer group must contain only trainable parameters"
        )

    optimizer_params = trainable_params
    if attention_scale_params:
        base_params = [
            param for param in trainable_params if id(param) not in scale_ids
        ]
        optimizer_params = []
        if base_params:
            optimizer_params.append(
                {
                    "params": base_params,
                    "lr": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "group_name": "model",
                }
            )
        scale_group = {
            "params": attention_scale_params,
            "lr": args.attention_scale_learning_rate,
            "weight_decay": 0.0,
            "group_name": "attention_scale",
        }
        if args.optimizer == "adamw":
            scale_group["eps"] = args.attention_scale_adam_eps
        optimizer_params.append(scale_group)
        grouped_ids = [
            id(param)
            for group in optimizer_params
            for param in group["params"]
        ]
        if (
            len(grouped_ids) != len(set(grouped_ids))
            or set(grouped_ids) != trainable_ids
        ):
            raise ValueError(
                "optimizer parameter groups must be disjoint and cover every "
                "trainable parameter exactly once"
            )
    if args.optimizer == "sgd":
        import torch

        return torch.optim.SGD(optimizer_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        import torch

        return torch.optim.AdamW(
            optimizer_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(float(getattr(args, "adam_beta1", 0.9)), float(getattr(args, "adam_beta2", 0.999))),
            eps=float(getattr(args, "adam_eps", 1e-8)),
        )
    if args.optimizer != "adafactor":
        raise ValueError(f"unsupported optimizer: {args.optimizer!r}")
    from transformers.optimization import Adafactor

    return Adafactor(
        optimizer_params,
        lr=args.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=args.weight_decay,
    )


def warmup_attention_scales(torch, model, scale_params, args) -> dict | None:
    """Optimize only scale controllers toward native temperature before training."""

    steps = args.attention_scale_prior_warmup_steps
    if steps <= 0:
        return None
    optimizer = torch.optim.AdamW(
        [
            {
                "params": scale_params,
                "lr": args.attention_scale_prior_warmup_learning_rate,
                "weight_decay": 0.0,
                "eps": args.attention_scale_adam_eps,
            }
        ]
    )
    initial_stats = collect_attention_scaling_stats(model)
    initial_loss = float(attention_scale_native_regularization(model).detach().cpu().item())
    best_loss = initial_loss
    best_step = 0
    best_state = [param.detach().clone() for param in scale_params]
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        regularization = attention_scale_native_regularization(model)
        regularization.backward()
        optimizer.step()
        project_attention_scales_(model, optimizer)
        candidate_loss = float(
            attention_scale_native_regularization(model).detach().cpu().item()
        )
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best_step = step
            best_state = [param.detach().clone() for param in scale_params]
    with torch.no_grad():
        for param, best in zip(scale_params, best_state, strict=True):
            param.copy_(best)
    if args.attention_scale_prior_warmup_snap_native:
        set_attention_scales_to_native_(model)
    final_stats = collect_attention_scaling_stats(model)
    final_loss = float(attention_scale_native_regularization(model).detach().cpu().item())
    print(
        "attention-scale prior warmup: "
        f"steps={steps} best_step={best_step} "
        f"regularization={initial_loss:.6g}->{final_loss:.6g}",
        flush=True,
    )
    return {
        "steps": steps,
        "learning_rate": args.attention_scale_prior_warmup_learning_rate,
        "best_step": best_step,
        "initial_regularization": initial_loss,
        "best_regularization": best_loss,
        "final_regularization": final_loss,
        "snapped_to_native": args.attention_scale_prior_warmup_snap_native,
        "initial_stats": initial_stats,
        "final_stats": final_stats,
    }


def _drop_unsupported_decoder_inputs(kwargs: dict):
    for key in ("token_type_ids",):
        kwargs.pop(key, None)
    return kwargs


def make_decoder_sequence_classifier(torch, nn, SequenceClassifierOutput, backbone, num_labels: int):
    class DecoderSequenceClassifier(nn.Module):
        def __init__(self, backbone, num_labels: int):
            super().__init__()
            self.backbone = backbone
            self.config = backbone.config
            self.num_labels = num_labels
            hidden_size = getattr(self.config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(self.config, "d_model", None)
            if hidden_size is None:
                raise ValueError(f"could not infer classifier hidden size from {type(self.config).__name__}")
            self.score = nn.Linear(hidden_size, num_labels, bias=False)
            backbone_dtype = next(backbone.parameters()).dtype
            self.score.to(dtype=backbone_dtype)

        def gradient_checkpointing_enable(self, *args, **kwargs):
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                return self.backbone.gradient_checkpointing_enable(*args, **kwargs)
            return None

        def enable_input_require_grads(self):
            if hasattr(self.backbone, "enable_input_require_grads"):
                return self.backbone.enable_input_require_grads()
            return None

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            kwargs = _drop_unsupported_decoder_inputs(kwargs)
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                **kwargs,
            )
            hidden = outputs.last_hidden_state
            batch_size = hidden.shape[0]
            if attention_mask is not None:
                sequence_lengths = attention_mask.to(torch.long).sum(dim=-1) - 1
                sequence_lengths = sequence_lengths.clamp(min=0)
            else:
                sequence_lengths = torch.full(
                    (batch_size,),
                    hidden.shape[1] - 1,
                    device=hidden.device,
                    dtype=torch.long,
                )
            pooled = hidden[torch.arange(batch_size, device=hidden.device), sequence_lengths]
            pooled = pooled.to(device=self.score.weight.device, dtype=self.score.weight.dtype)
            logits = self.score(pooled)
            loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                problem_type = getattr(self.config, "problem_type", None)
                if problem_type is None:
                    if self.num_labels == 1:
                        problem_type = "regression"
                    elif labels.dtype in (torch.long, torch.int):
                        problem_type = "single_label_classification"
                    else:
                        problem_type = "multi_label_classification"
                    self.config.problem_type = problem_type
                if problem_type == "regression":
                    loss = torch.nn.functional.mse_loss(logits.squeeze(), labels.squeeze())
                elif problem_type == "single_label_classification":
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
                else:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=getattr(outputs, "hidden_states", None),
                attentions=getattr(outputs, "attentions", None),
            )

    return DecoderSequenceClassifier(backbone, num_labels)


def load_model(torch, args, tokenizer, device):
    from torch import nn
    from transformers import AutoModel, AutoModelForSequenceClassification
    from transformers.modeling_outputs import SequenceClassifierOutput

    kwargs = {
        "cache_dir": args.cache_dir,
        "local_files_only": args.offline,
        "torch_dtype": torch_dtype(torch, args.dtype),
        "low_cpu_mem_usage": True,
    }
    if getattr(args, "trainable_attention_scaling", False):
        # OLMo and Mistral pass this value to the attention implementation.
        # Eager attention accepts a differentiable Tensor; SDPA/Flash require
        # a Python float and would detach or reject the learned scale.
        kwargs["attn_implementation"] = "eager"
    if args.device_map is not None:
        kwargs["device_map"] = args.device_map
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch_dtype(torch, args.bnb_4bit_compute_dtype or args.dtype)
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = None
    hf_loader = None
    if not args.force_decoder_wrapper:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model,
                num_labels=glue_num_labels(args.task),
                **kwargs,
            )
            hf_loader = "transformers.AutoModelForSequenceClassification"
        except Exception as exc:
            print(
                f"AutoModelForSequenceClassification failed for {args.model}; "
                f"falling back to AutoModel classifier wrapper: {exc}",
                flush=True,
            )
    if model is None:
        backbone = AutoModel.from_pretrained(args.model, **kwargs)
        model = make_decoder_sequence_classifier(
            torch,
            nn,
            SequenceClassifierOutput,
            backbone,
            glue_num_labels(args.task),
        )
        hf_loader = "transformers.AutoModel"
    model._qft_huggingface_loader = hf_loader
    if args.task == "stsb":
        model.config.problem_type = "regression"
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if args.device_map is None:
        model.to(device)
    elif hasattr(model, "score"):
        first_param = next(model.backbone.parameters())
        model.score.to(device=first_param.device)
    return model


def apply_peft_lora_if_requested(model, args):
    if not args.peft_lora:
        return model
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    target_modules = args.lora_target_modules or [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "w1",
        "w2",
        "w3",
    ]
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_modules,
    )
    if hasattr(model, "backbone"):
        model.backbone = prepare_model_for_kbit_training(model.backbone, use_gradient_checkpointing=False)
        model.backbone = get_peft_model(model.backbone, config)
        if hasattr(model.backbone, "print_trainable_parameters"):
            model.backbone.print_trainable_parameters()
        return model
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def train_variant(torch, args, tokenizer, cap, metric, target, train_loader, eval_loader, device):
    from transformers import (
        get_constant_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )

    seed_everything(args.seed)
    model = load_model(torch, args, tokenizer, device)
    model = apply_peft_lora_if_requested(model, args)
    capped = cap is not None
    applied_caps = 0
    before_stats = fast_weight_stats(model)
    if capped:
        applied_caps = apply_learnable_frobenius_caps(
            model,
            cap,
            norm_dtype=args.cap_norm_dtype,
        )
        if args.device_map is None:
            model.to(device)
    after_cap_stats = fast_weight_stats(model)
    alpha_s_calibration = None
    effective_alpha_s = args.alpha_s
    if args.attention_scaling == "alpha_s_alpha_w" and args.alpha_s_calibration_batches:
        alpha_s_calibration = calibrate_attention_input_bound(
            torch,
            model,
            eval_loader,
            device,
            args.alpha_s_calibration_batches,
        )
        calibrated_bound = (
            alpha_s_calibration["observed_max_token_l2"]
            * args.alpha_s_calibration_margin
        )
        effective_alpha_s = max(float(args.alpha_s), calibrated_bound)
        alpha_s_calibration["margin"] = args.alpha_s_calibration_margin
        alpha_s_calibration["configured_alpha_s"] = args.alpha_s
        alpha_s_calibration["effective_alpha_s"] = effective_alpha_s
        print(
            "alpha_s calibration: "
            f"observed={alpha_s_calibration['observed_max_token_l2']:.6g} "
            f"margin={args.alpha_s_calibration_margin:g} "
            f"effective={effective_alpha_s:.6g}",
            flush=True,
        )
    attention_scaling_layers = apply_attention_scaling(
        model,
        args.attention_scaling,
        sequence_length=args.max_length if args.attention_scaling == "sequence_length" else None,
        alpha_s=effective_alpha_s if args.attention_scaling == "alpha_s_alpha_w" else None,
        alpha_w=args.alpha_w if args.attention_scaling == "alpha_s_alpha_w" else None,
        trainable=args.trainable_attention_scaling,
        shared_trainable=args.shared_attention_scale,
        enforce_alpha_denominator_bound=args.enforce_alpha_denominator_bound,
        min_score_scale=args.attention_scale_min,
        max_score_scale=args.attention_scale_max,
    )
    configure_trainable(
        model,
        args.train_scope,
        capped=capped,
        peft_lora=args.peft_lora,
        trainable_attention_scaling=args.trainable_attention_scaling,
    )
    attention_scale_controller_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and "_qft_learnable_attention_scale." in name
    ]
    attention_scale_warmup = warmup_attention_scales(
        torch,
        model,
        attention_scale_controller_params,
        args,
    )
    if args.freeze_attention_scale_after_prior_warmup:
        for param in attention_scale_controller_params:
            param.requires_grad = False
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    trainable_params_count = sum(param.numel() for param in trainable_params)
    attention_scale_params = [
        param for param in attention_scale_controller_params if param.requires_grad
    ]
    attention_scale_param_ids = {id(param) for param in attention_scale_params}
    base_trainable_params = [
        param for param in trainable_params if id(param) not in attention_scale_param_ids
    ]
    optimizer = make_optimizer(
        args,
        trainable_params,
        attention_scale_params,
    )
    checkpoints = sorted(set(max(0, step) for step in args.checkpoints))
    max_steps = max(checkpoints)
    warmup_steps = int(args.warmup_ratio * max_steps)
    if max_steps <= 0:
        scheduler = None
    elif args.lr_scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            warmup_steps,
            max_steps,
        )

    rows = []
    losses = []
    attention_scale_regularization_losses = []
    model_grad_norms = []
    attention_scale_grad_norms = []
    start = time.time()
    variant_name = "baseline" if cap is None else f"fnlearn:{cap:g}"
    kind = "baseline" if cap is None else "fnlearn"
    saved_recovery_checkpoint = None

    def save_recovery_checkpoint(step: int) -> str | None:
        nonlocal saved_recovery_checkpoint
        if not args.save_recovery_checkpoint:
            return None
        if saved_recovery_checkpoint is not None:
            return saved_recovery_checkpoint
        output = Path(args.output)
        cap_label = "baseline" if cap is None else f"fn{cap:g}"
        path = output.with_name(f"{output.stem}.{cap_label}.recovery.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        trainable_state = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad or "_qft_learnable_attention_scale." in name
        }
        checkpoint_payload = {
            "format": (
                "qft_alpha_bounded_trainable_state_v2"
                if args.enforce_alpha_denominator_bound
                else "qft_trainable_state_v1"
            ),
            "step": step,
            "model": args.model,
            "task": args.task,
            "cap": cap,
            "train_scope": args.train_scope,
            "learning_rate": args.learning_rate,
            "attention_scaling": args.attention_scaling,
            "trainable_attention_scaling": args.trainable_attention_scaling,
            "shared_attention_scale": args.shared_attention_scale,
            "enforce_alpha_denominator_bound": (
                args.enforce_alpha_denominator_bound
            ),
            "attention_scale_controller": (
                "AlphaBoundedAttentionScale.log_denominator_slack"
                if args.enforce_alpha_denominator_bound
                else "LearnableAttentionScale.raw_scale"
            ),
            "attention_scale_native_regularization": (
                args.attention_scale_native_regularization
            ),
            "attention_scale_warmup": attention_scale_warmup,
            "freeze_attention_scale_after_prior_warmup": (
                args.freeze_attention_scale_after_prior_warmup
            ),
            "alpha_s": effective_alpha_s,
            "configured_alpha_s": args.alpha_s,
            "alpha_w": args.alpha_w,
            "attention_scale_min": args.attention_scale_min,
            "attention_scale_max": args.attention_scale_max,
            "attention_scale_stats": collect_attention_scaling_stats(model),
            "reconstruction_order": [
                "base_model",
                "peft_if_configured",
                "frobenius_caps",
                "attention_scaling_adapter",
            ],
            "trainable_state_dict": trainable_state,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint_payload, temporary)
        temporary.replace(path)
        saved_recovery_checkpoint = str(path)
        print(f"saved recovered trainable state to {path}", flush=True)
        return saved_recovery_checkpoint

    def evaluate_checkpoint(step):
        metrics = evaluate(torch, model, eval_loader, device, args.task)
        value = metrics.get(metric)
        caps = collect_learnable_cap_stats(model) if capped else []
        scales = collect_attention_scaling_stats(model)
        recovered = meets_threshold(value, target, metric, args.tolerance)
        checkpoint_path = save_recovery_checkpoint(step) if recovered is True else None
        rows.append(
            {
                "variant": variant_name,
                "kind": kind,
                "cap": cap,
                "checkpoint_steps": step,
                "metric": metric,
                "metric_value": value,
                "baseline_target": target,
                "threshold": threshold_value(target, metric, args.tolerance),
                "meets_threshold": recovered,
                "mean_train_loss": (sum(losses) / len(losses)) if losses else None,
                "mean_attention_scale_native_regularization": (
                    sum(attention_scale_regularization_losses)
                    / len(attention_scale_regularization_losses)
                    if attention_scale_regularization_losses
                    else None
                ),
                "mean_model_grad_norm": (
                    sum(model_grad_norms) / len(model_grad_norms)
                    if model_grad_norms
                    else None
                ),
                "mean_attention_scale_grad_norm": (
                    sum(attention_scale_grad_norms) / len(attention_scale_grad_norms)
                    if attention_scale_grad_norms
                    else None
                ),
                "runtime_sec": time.time() - start,
                "metrics": metrics,
                "weight_stats_final": fast_weight_stats(model),
                "cap_stats": caps,
                "cap_summary": cap_summary(caps),
                "attention_scale_stats": scales,
                "attention_scale_summary": attention_scale_summary(scales),
                "recovery_checkpoint": checkpoint_path,
            }
        )
        print(
            f"{variant_name} step={step} {metric}={value:.6g} "
            f"threshold={threshold_value(target, metric, args.tolerance)} "
            f"meets={recovered}",
            flush=True,
        )
        return recovered is True

    recovered = False
    if 0 in checkpoints:
        recovered = evaluate_checkpoint(0)

    step_count = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    while step_count < max_steps and not (args.early_stop_on_recovery and recovered):
        for batch in train_loader:
            batch = move_batch(batch, device)
            outputs = call_model(model, batch)
            task_loss = outputs.loss
            scale_regularization = None
            if args.attention_scale_native_regularization > 0:
                scale_regularization = attention_scale_native_regularization(model)
                task_loss_with_regularization = (
                    task_loss
                    + args.attention_scale_native_regularization * scale_regularization
                )
            else:
                task_loss_with_regularization = task_loss
            loss = task_loss_with_regularization / args.grad_accum_steps
            loss.backward()
            micro_step += 1
            if micro_step % args.grad_accum_steps != 0:
                continue
            if args.max_grad_norm and args.max_grad_norm > 0:
                if base_trainable_params:
                    model_grad_norm = torch.nn.utils.clip_grad_norm_(
                        base_trainable_params,
                        args.max_grad_norm,
                    )
                    model_grad_norms.append(
                        float(model_grad_norm.detach().float().cpu().item())
                    )
                if attention_scale_params:
                    scale_grad_norm = torch.nn.utils.clip_grad_norm_(
                        attention_scale_params,
                        args.max_grad_norm,
                    )
                    attention_scale_grad_norms.append(
                        float(scale_grad_norm.detach().float().cpu().item())
                    )
            optimizer.step()
            project_attention_scales_(model, optimizer)
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(task_loss.detach().cpu().item()))
            if scale_regularization is not None:
                attention_scale_regularization_losses.append(
                    float(scale_regularization.detach().cpu().item())
                )
            step_count += 1
            if step_count in checkpoints:
                recovered = evaluate_checkpoint(step_count)
                model.train()
                if args.early_stop_on_recovery and recovered:
                    break
            if step_count >= max_steps:
                break

    result = {
        "variant": variant_name,
        "kind": kind,
        "cap": cap,
        "applied_caps": applied_caps,
        "attention_scaling_layers": attention_scaling_layers,
        "attention_scale_warmup": attention_scale_warmup,
        "attention_scale_frozen_after_warmup": (
            args.freeze_attention_scale_after_prior_warmup
        ),
        "alpha_denominator_bound_enforced": args.enforce_alpha_denominator_bound,
        "alpha_s_calibration": alpha_s_calibration,
        "effective_alpha_s": effective_alpha_s,
        "attention_scale_stats_final": collect_attention_scaling_stats(model),
        "recovered": any(row.get("meets_threshold") is True for row in rows),
        "stopped_early": bool(
            args.early_stop_on_recovery
            and any(row.get("meets_threshold") is True for row in rows)
            and step_count < max_steps
        ),
        "recovery_checkpoint": saved_recovery_checkpoint,
        "trainable_params": trainable_params_count,
        "model_loading": {
            "loader": getattr(model, "_qft_huggingface_loader", None),
            "model_class": type(model).__name__,
            "backbone_class": (
                type(model.backbone).__name__ if hasattr(model, "backbone") else None
            ),
            "attention_implementation": getattr(
                getattr(model, "config", None), "_attn_implementation", None
            ),
            "frobenius_cap_adapter": (
                "quantum_friendly_transformer.norm_transformer."
                "apply_learnable_frobenius_caps"
            ),
            "attention_scaling_adapter": (
                "quantum_friendly_transformer.norm_transformer."
                "apply_attention_scaling"
            ),
        },
        "weight_stats_before": before_stats,
        "weight_stats_after_cap": after_cap_stats,
        "checkpoints": rows,
    }
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def summarize_minimal(results: list[dict]) -> dict:
    summary = []
    for result in results:
        successes = [row for row in result["checkpoints"] if row.get("meets_threshold") is True]
        first = min(successes, key=lambda row: row["checkpoint_steps"]) if successes else None
        summary.append(
            {
                "cap": result["cap"],
                "minimal_recovery_steps": first["checkpoint_steps"] if first else None,
                "minimal_metric_value": first["metric_value"] if first else None,
                "recovered": first is not None,
            }
        )
    recovered = [row for row in summary if row["recovered"]]
    return {
        "by_cap": summary,
        "minimal_cap_any_recovery": min(recovered, key=lambda row: (row["cap"], row["minimal_recovery_steps"])) if recovered else None,
        "minimal_steps_any_cap": min(recovered, key=lambda row: (row["minimal_recovery_steps"], row["cap"])) if recovered else None,
    }


def main() -> None:
    args = parse_args()
    import os
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    seed_everything(args.seed)
    validate_attention_scaling_args(args)
    validate_optimizer_args(args)
    if args.attention_scaling == "sequence_length" and not args.pad_to_max_length:
        raise ValueError("sequence_length attention scaling requires --pad-to-max-length so N is fixed")
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
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_ds, eval_ds = prepare_dataset(args, tokenizer, load_dataset)
    train_loader, eval_loader = make_loaders(torch, args, tokenizer, train_ds, eval_ds)

    metric = args.metric or primary_metric(args.task)
    explicit_target = args.baseline_target if args.baseline_target is not None else baseline_target_from_json(args.baseline_json, metric)
    caps = args.caps or []
    if not args.include_baseline and not caps:
        raise ValueError("provide --caps, or use --include-baseline for a baseline-only run")

    baseline_run = None
    target = explicit_target
    if args.include_baseline:
        print(f"running baseline on {args.task}", flush=True)
        baseline_run = train_variant(torch, args, tokenizer, None, metric, None, train_loader, eval_loader, device)
        target = select_baseline_target(baseline_run["checkpoints"], metric, args.baseline_selection)

    results = []
    for cap in caps:
        print(f"running 7B learnable FN cap {cap:g} on {args.task}", flush=True)
        results.append(train_variant(torch, args, tokenizer, cap, metric, target, train_loader, eval_loader, device))

    output = ensure_parent(args.output)
    payload = {
        **now_metadata(),
        "model": args.model,
        "task": args.task,
        "dataset": "glue",
        "split": glue_split_name(args.task, "validation"),
        "seed": args.seed,
        "device": str(device),
        "cache_dir": args.cache_dir,
        "offline": args.offline,
        "caps": caps,
        "checkpoints": sorted(set(args.checkpoints)),
        "metric": metric,
        "baseline_target": target,
        "baseline_selection": args.baseline_selection,
        "tolerance": args.tolerance,
        "threshold": threshold_value(target, metric, args.tolerance),
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler": args.lr_scheduler,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "adam_beta1": float(getattr(args, "adam_beta1", 0.9)),
        "adam_beta2": float(getattr(args, "adam_beta2", 0.999)),
        "adam_eps": float(getattr(args, "adam_eps", 1e-8)),
        "max_grad_norm": args.max_grad_norm,
        "train_scope": args.train_scope,
        "dtype": args.dtype,
        "cap_norm_dtype": args.cap_norm_dtype,
        "gradient_checkpointing": args.gradient_checkpointing,
        "device_map": args.device_map,
        "force_decoder_wrapper": args.force_decoder_wrapper,
        "load_in_4bit": args.load_in_4bit,
        "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
        "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype or args.dtype,
        "peft_lora": args.peft_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "max_length": args.max_length,
        "attention_scaling": args.attention_scaling,
        "trainable_attention_scaling": args.trainable_attention_scaling,
        "shared_attention_scale": args.shared_attention_scale,
        "enforce_alpha_denominator_bound": args.enforce_alpha_denominator_bound,
        "attention_scale_learning_rate": args.attention_scale_learning_rate,
        "attention_scale_adam_eps": args.attention_scale_adam_eps,
        "attention_scale_native_regularization": args.attention_scale_native_regularization,
        "attention_scale_prior_warmup_steps": args.attention_scale_prior_warmup_steps,
        "attention_scale_prior_warmup_learning_rate": (
            args.attention_scale_prior_warmup_learning_rate
        ),
        "attention_scale_prior_warmup_snap_native": (
            args.attention_scale_prior_warmup_snap_native
        ),
        "freeze_attention_scale_after_prior_warmup": (
            args.freeze_attention_scale_after_prior_warmup
        ),
        "attention_scale_min": args.attention_scale_min,
        "attention_scale_max": args.attention_scale_max,
        "alpha_s_calibration_batches": args.alpha_s_calibration_batches,
        "alpha_s_calibration_margin": args.alpha_s_calibration_margin,
        "alpha_s": args.alpha_s,
        "alpha_w": args.alpha_w,
        "attention_scale_denominator": attention_scale_denominator(args),
        "pad_to_max_length": args.pad_to_max_length,
        "tensor_sequence_length": args.max_length if args.pad_to_max_length else None,
        "early_stop_on_recovery": args.early_stop_on_recovery,
        "save_recovery_checkpoint": args.save_recovery_checkpoint,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "package_versions": package_versions(),
        "baseline_run": baseline_run,
        "results": results,
        "minimal_recovery": summarize_minimal(results),
    }
    write_json(output, payload)
    write_json(metadata_path(output), {k: v for k, v in payload.items() if k not in {"results", "baseline_run"}})
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
