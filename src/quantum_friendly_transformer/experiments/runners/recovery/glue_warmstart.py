#!/usr/bin/env python
"""Warm-start GLUE recovery after applying a learnable Frobenius cap."""

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
    cap_summary,
    configure_trainable,
    evaluate,
    fast_weight_stats,
    load_model,
    make_loaders,
    make_optimizer,
    call_model,
    meets_threshold,
    pick_device,
    prepare_dataset,
    primary_metric,
    select_baseline_target,
    threshold_value,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=sorted(GLUE_TEXT_COLUMNS))
    parser.add_argument("--cap", type=float, required=True)
    parser.add_argument(
        "--anneal-caps",
        type=float,
        nargs="+",
        default=None,
        help="Optional decreasing cap schedule ending at --cap, e.g. 64 48 32 24.",
    )
    parser.add_argument(
        "--anneal-stage-steps",
        type=int,
        nargs="+",
        default=None,
        help="Training steps for each non-final annealing cap.",
    )
    parser.add_argument(
        "--anneal-checkpoint-interval",
        type=int,
        default=0,
        help="If positive, evaluate non-final annealing stages every N optimizer steps and carry forward the best checkpoint.",
    )
    parser.add_argument("--baseline-checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--recovery-checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--baseline-target", type=float, default=None)
    parser.add_argument("--baseline-selection", choices=["final", "best"], default="best")
    parser.add_argument("--warm-start-selection", choices=["final", "best"], default="best")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--pad-to-max-length", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--baseline-learning-rate", type=float, default=2e-5)
    parser.add_argument("--recovery-learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--anneal-stage-learning-rates",
        type=float,
        nargs="+",
        default=None,
        help="Optional learning rate per non-final annealing cap.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--baseline-warmup-ratio", type=float, default=0.06)
    parser.add_argument("--recovery-warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--baseline-train-scope", choices=["head_caps", "head_pooler_caps", "target_caps", "all"], default="all")
    parser.add_argument("--recovery-train-scope", choices=["head_caps", "head_pooler_caps", "target_caps", "all"], default="all")
    parser.add_argument(
        "--anneal-stage-train-scopes",
        choices=["head_caps", "head_pooler_caps", "target_caps", "all"],
        nargs="+",
        default=None,
        help="Optional train scope per non-final annealing cap.",
    )
    parser.add_argument("--optimizer", choices=["sgd", "adamw", "adafactor"], default="adamw")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="float32")
    parser.add_argument("--cap-norm-dtype", choices=["native", "float32"], default="float32")
    parser.add_argument("--cap-init-fraction", type=float, default=16.0 / 17.0)
    parser.add_argument("--freeze-cap-scalars", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def clone_state_dict_cpu(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def is_better(metric: str, candidate: dict, incumbent: dict | None) -> bool:
    if incumbent is None:
        return True
    if metric == "eval_loss":
        return candidate["metric_value"] < incumbent["metric_value"]
    return candidate["metric_value"] > incumbent["metric_value"]


def optimizer_args(args: argparse.Namespace, lr: float, train_scope: str) -> argparse.Namespace:
    return argparse.Namespace(
        optimizer=args.optimizer,
        learning_rate=lr,
        weight_decay=args.weight_decay,
        train_scope=train_scope,
    )


def cap_schedule(args: argparse.Namespace) -> list[float]:
    caps = args.anneal_caps or [args.cap]
    if not caps:
        raise ValueError("cap schedule is empty")
    if abs(caps[-1] - args.cap) > 1e-9:
        raise ValueError("--anneal-caps must end at --cap")
    if any(next_cap > cap for cap, next_cap in zip(caps, caps[1:])):
        raise ValueError("--anneal-caps must be non-increasing")
    if len(caps) > 1:
        expected = len(caps) - 1
        if args.anneal_stage_steps is None or len(args.anneal_stage_steps) != expected:
            raise ValueError("--anneal-stage-steps must have one entry per non-final annealing cap")
        if args.anneal_stage_learning_rates is not None and len(args.anneal_stage_learning_rates) != expected:
            raise ValueError("--anneal-stage-learning-rates must have one entry per non-final annealing cap")
        if args.anneal_stage_train_scopes is not None and len(args.anneal_stage_train_scopes) != expected:
            raise ValueError("--anneal-stage-train-scopes must have one entry per non-final annealing cap")
    return caps


def set_learnable_cap_max(model, max_gamma: float, init_fraction: float | None = None) -> int:
    import torch
    from torch.nn.utils import parametrize

    updated = 0
    for _, module in target_weight_modules(model):
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


def stage_checkpoints(stage_steps: int, interval: int) -> list[int]:
    if interval <= 0:
        return [0, stage_steps]
    checkpoints = list(range(0, stage_steps + 1, interval))
    if checkpoints[-1] != stage_steps:
        checkpoints.append(stage_steps)
    return checkpoints


def train_phase(
    torch,
    args: argparse.Namespace,
    model,
    train_loader,
    eval_loader,
    device,
    *,
    phase: str,
    cap: float | None,
    metric: str,
    target: float | None,
    checkpoints: list[int],
    learning_rate: float,
    warmup_ratio: float,
    train_scope: str,
    save_selection: str | None = None,
) -> tuple[dict, dict | None]:
    from transformers import get_linear_schedule_with_warmup

    capped = cap is not None
    configure_trainable(model, train_scope, capped=capped)
    if capped and args.freeze_cap_scalars:
        for name, param in model.named_parameters():
            if "parametrizations.weight.0.g" in name.lower():
                param.requires_grad = False
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    trainable_params_count = sum(param.numel() for param in trainable_params)
    optimizer = make_optimizer(optimizer_args(args, learning_rate, train_scope), trainable_params)
    checkpoints = sorted(set(max(0, step) for step in checkpoints))
    max_steps = max(checkpoints)
    warmup_steps = int(warmup_ratio * max_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max_steps) if max_steps > 0 else None

    rows = []
    losses = []
    start = time.time()
    selected_row = None
    selected_state = None
    variant_name = "baseline" if cap is None else f"fnlearn:{cap:g}:warmstart"
    kind = "baseline" if cap is None else "fnlearn"

    def evaluate_checkpoint(step: int):
        nonlocal selected_row, selected_state
        metrics = evaluate(torch, model, eval_loader, device, args.task)
        value = metrics.get(metric)
        caps = collect_learnable_cap_stats(model) if capped else []
        row = {
            "variant": variant_name,
            "kind": kind,
            "phase": phase,
            "cap": cap,
            "checkpoint_steps": step,
            "metric": metric,
            "metric_value": value,
            "baseline_target": target,
            "threshold": threshold_value(target, metric, args.tolerance),
            "meets_threshold": meets_threshold(value, target, metric, args.tolerance),
            "mean_train_loss": (sum(losses) / len(losses)) if losses else None,
            "runtime_sec": time.time() - start,
            "metrics": metrics,
            "weight_stats_final": fast_weight_stats(model),
            "cap_stats": caps,
            "cap_summary": cap_summary(caps),
        }
        rows.append(row)
        if save_selection == "best" and is_better(metric, row, selected_row):
            selected_row = row
            selected_state = clone_state_dict_cpu(model)
        print(
            f"{variant_name} {phase} step={step} {metric}={value:.6g} "
            f"threshold={threshold_value(target, metric, args.tolerance)} "
            f"meets={meets_threshold(value, target, metric, args.tolerance)}",
            flush=True,
        )

    if 0 in checkpoints:
        evaluate_checkpoint(0)

    step_count = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    while step_count < max_steps:
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = call_model(model, batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            micro_step += 1
            if micro_step % args.grad_accum_steps != 0:
                continue
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float((loss.detach() * args.grad_accum_steps).cpu().item()))
            step_count += 1
            if step_count in checkpoints:
                evaluate_checkpoint(step_count)
                model.train()
            if step_count >= max_steps:
                break

    if save_selection == "final" and rows:
        selected_row = rows[-1]

    result = {
        "variant": variant_name,
        "kind": kind,
        "phase": phase,
        "cap": cap,
        "trainable_params": trainable_params_count,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "train_scope": train_scope,
        "checkpoints": rows,
        "selected_checkpoint": selected_row,
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
        vocab = tokenizer.get_vocab()
        for candidate in (tokenizer.eos_token, tokenizer.bos_token, tokenizer.unk_token, "<pad>", "</s>", "<s>"):
            if candidate is not None and candidate in vocab:
                tokenizer.pad_token = candidate
                break
        if tokenizer.pad_token is None:
            raise ValueError(f"tokenizer for {args.model} has no usable existing pad token")
    tokenizer.padding_side = "right"
    train_ds, eval_ds = prepare_dataset(args, tokenizer, load_dataset)
    train_loader, eval_loader = make_loaders(torch, args, tokenizer, train_ds, eval_ds)

    metric = args.metric or primary_metric(args.task)
    schedule = cap_schedule(args)
    model = load_model(torch, args, tokenizer, device)
    weight_stats_before = fast_weight_stats(model)

    print(f"running warm-start baseline on {args.task}", flush=True)
    baseline_run, selected_state = train_phase(
        torch,
        args,
        model,
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
    if selected_state is not None:
        model.load_state_dict(selected_state)
        model.to(device)

    target = args.baseline_target
    if target is None:
        target = select_baseline_target(baseline_run["checkpoints"], metric, args.baseline_selection)

    print(f"applying learnable FN cap {schedule[0]:g}", flush=True)
    applied_caps = apply_learnable_frobenius_caps(
        model,
        schedule[0],
        init_fraction=args.cap_init_fraction,
        norm_dtype=args.cap_norm_dtype,
    )
    model.to(device)
    weight_stats_after_cap = fast_weight_stats(model)

    recovery_runs = []
    for stage_index, stage_cap in enumerate(schedule):
        if stage_index > 0:
            updated = set_learnable_cap_max(model, stage_cap, init_fraction=args.cap_init_fraction)
            print(f"lowered learnable FN cap to {stage_cap:g} on {updated} parametrizations", flush=True)
        final_stage = stage_index == len(schedule) - 1
        if final_stage:
            checkpoints = args.recovery_checkpoints
            phase = "recovery"
            stage_learning_rate = args.recovery_learning_rate
        else:
            checkpoints = stage_checkpoints(args.anneal_stage_steps[stage_index], args.anneal_checkpoint_interval)
            phase = f"anneal_cap_{stage_cap:g}"
            stage_learning_rate = (
                args.anneal_stage_learning_rates[stage_index]
                if args.anneal_stage_learning_rates is not None
                else args.recovery_learning_rate
            )
        stage_train_scope = (
            args.anneal_stage_train_scopes[stage_index]
            if (not final_stage and args.anneal_stage_train_scopes is not None)
            else args.recovery_train_scope
        )
        print(f"running warm-start {phase} cap {stage_cap:g} on {args.task}", flush=True)
        recovery_run, selected_stage_state = train_phase(
            torch,
            args,
            model,
            train_loader,
            eval_loader,
            device,
            phase=phase,
            cap=stage_cap,
            metric=metric,
            target=target,
            checkpoints=checkpoints,
            learning_rate=stage_learning_rate,
            warmup_ratio=args.recovery_warmup_ratio,
            train_scope=stage_train_scope,
            save_selection=None if final_stage else "best",
        )
        recovery_runs.append(recovery_run)
        if not final_stage and selected_stage_state is not None:
            model.load_state_dict(selected_stage_state)
            model.to(device)
    recovery_run = recovery_runs[-1]

    output = ensure_parent(args.output)
    payload = {
        **now_metadata(),
        "model": args.model,
        "task": args.task,
        "dataset": "glue",
        "seed": args.seed,
        "device": str(device),
        "cache_dir": args.cache_dir,
        "offline": args.offline,
        "cap": args.cap,
        "anneal_caps": schedule,
        "anneal_stage_steps": args.anneal_stage_steps,
        "anneal_stage_learning_rates": args.anneal_stage_learning_rates,
        "anneal_stage_train_scopes": args.anneal_stage_train_scopes,
        "anneal_checkpoint_interval": args.anneal_checkpoint_interval,
        "metric": metric,
        "baseline_target": target,
        "baseline_selection": args.baseline_selection,
        "warm_start_selection": args.warm_start_selection,
        "tolerance": args.tolerance,
        "threshold": threshold_value(target, metric, args.tolerance),
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "baseline_learning_rate": args.baseline_learning_rate,
        "recovery_learning_rate": args.recovery_learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "baseline_train_scope": args.baseline_train_scope,
        "recovery_train_scope": args.recovery_train_scope,
        "dtype": args.dtype,
        "cap_norm_dtype": args.cap_norm_dtype,
        "cap_init_fraction": args.cap_init_fraction,
        "freeze_cap_scalars": args.freeze_cap_scalars,
        "gradient_checkpointing": args.gradient_checkpointing,
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
    write_json(output, payload)
    write_json(metadata_path(output), {k: v for k, v in payload.items() if k not in {"baseline_run", "recovery_run", "recovery_runs"}})
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
