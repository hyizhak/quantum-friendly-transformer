import argparse
import gc
import time

from quantum_friendly_transformer.norm_transformer.frobenius_cap import (
    apply_learnable_frobenius_caps,
    collect_learnable_cap_stats,
    collect_weight_stats,
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
from quantum_friendly_transformer.experiments.runners.normalisation.wikitext import chunk_records, evaluate, make_loader, move_batch, pick_device


LOWER_IS_BETTER = {"loss", "perplexity"}


def parse_args():
    parser = argparse.ArgumentParser(description="Find recovery checkpoints for WikiText learnable-cap FN models.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--config", default="wikitext-103-raw-v1")
    parser.add_argument("--caps", type=float, nargs="+", required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--metric", default="perplexity", choices=["loss", "perplexity"])
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--baseline-target", type=float, default=None)
    parser.add_argument("--baseline-selection", choices=["final", "best"], default="final")
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--absolute-tolerance", type=float, default=None)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--max-eval-tokens", type=int, default=None)
    parser.add_argument("--train-scope", choices=["all", "target_caps", "caps_only"], default="all")
    parser.add_argument("--optimizer", choices=["adamw", "sgd", "adafactor"], default="adamw")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--cap-norm-dtype", choices=["float32", "native"], default="float32")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--early-stop-on-recovery", action="store_true")
    parser.add_argument("--stop-after-first-recovered", action="store_true")
    parser.add_argument("--fast-weight-stats", action="store_true")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def baseline_target(args):
    if args.baseline_target is not None:
        return args.baseline_target
    if not args.baseline_json:
        return None
    import json
    from pathlib import Path

    payload = json.loads(Path(args.baseline_json).read_text())
    if payload.get("baseline_target") is not None:
        return float(payload["baseline_target"])
    baseline_run = payload.get("baseline_run")
    if baseline_run and baseline_run.get("checkpoints"):
        return float(baseline_run["checkpoints"][-1]["metric_value"])
    candidates = [row for row in payload.get("results", []) if row.get("variant") == "baseline"]
    if not candidates:
        raise ValueError(f"no baseline result found in {args.baseline_json}")
    value = candidates[0].get("metrics", {}).get(args.metric)
    if value is None:
        raise ValueError(f"metric {args.metric!r} missing from baseline in {args.baseline_json}")
    return float(value)


def threshold_value(target, args):
    if target is None:
        return None
    if args.absolute_tolerance is not None:
        if args.metric in LOWER_IS_BETTER:
            return target + args.absolute_tolerance
        return target - args.absolute_tolerance
    if args.metric in LOWER_IS_BETTER:
        return target * (1.0 + args.relative_tolerance)
    return target * (1.0 - args.relative_tolerance)


def meets_threshold(value, target, args):
    if target is None or value is None:
        return None
    threshold = threshold_value(target, args)
    if args.metric in LOWER_IS_BETTER:
        return value <= threshold
    return value >= threshold


def select_baseline_target(checkpoints, metric, selection):
    if not checkpoints:
        raise ValueError("baseline run has no checkpoints")
    if selection == "final":
        return float(checkpoints[-1]["metric_value"])
    if metric in LOWER_IS_BETTER:
        return float(min(checkpoints, key=lambda row: row["metric_value"])["metric_value"])
    return float(max(checkpoints, key=lambda row: row["metric_value"])["metric_value"])


def cap_summary(cap_stats):
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


def config_metadata(config):
    return {
        "model_type": getattr(config, "model_type", None),
        "hidden_dim": getattr(config, "hidden_size", None) or getattr(config, "n_embd", None),
        "num_layers": getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None),
        "num_heads": getattr(config, "num_attention_heads", None) or getattr(config, "n_head", None),
    }


def torch_dtype(torch, dtype):
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    return torch.float32


def fast_weight_stats(model):
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


def weight_stats(model, fast):
    return fast_weight_stats(model) if fast else collect_weight_stats(model)


def configure_trainable(model, train_scope, capped):
    for param in model.parameters():
        param.requires_grad = train_scope == "all"
    if train_scope == "all":
        return sum(param.numel() for param in model.parameters() if param.requires_grad)

    for name, param in model.named_parameters():
        lowered = name.lower()
        if "parametrizations.weight.0.g" in lowered:
            param.requires_grad = True
        elif train_scope == "target_caps" and "parametrizations.weight.original" in lowered:
            param.requires_grad = True
        elif param.ndim == 1 and ("norm" in lowered or "ln_" in lowered or "layer_norm" in lowered):
            param.requires_grad = True

    if train_scope == "target_caps" and not capped:
        target_param_ids = {id(module.weight) for _, module in target_weight_modules(model)}
        for param in model.parameters():
            if id(param) in target_param_ids:
                param.requires_grad = True

    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def make_optimizer(torch, args, trainable_params):
    if not trainable_params:
        raise ValueError("no trainable parameters selected")
    if args.optimizer == "sgd":
        return torch.optim.SGD(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    from transformers.optimization import Adafactor

    return Adafactor(
        trainable_params,
        lr=args.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=args.weight_decay,
    )


def load_model(torch, args, tokenizer, device):
    from transformers import AutoModelForCausalLM

    kwargs = {
        "cache_dir": args.cache_dir,
        "local_files_only": args.offline,
    }
    if args.dtype != "float32":
        kwargs["torch_dtype"] = torch_dtype(torch, args.dtype)
    if args.low_cpu_mem_usage:
        kwargs["low_cpu_mem_usage"] = True
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.to(device)
    return model


def train_variant(torch, args, tokenizer, cap, target, train_loader, eval_loader, device):
    from transformers import get_linear_schedule_with_warmup

    seed_everything(args.seed)
    model = load_model(torch, args, tokenizer, device)
    capped = cap is not None
    applied_caps = 0
    before_stats = weight_stats(model, args.fast_weight_stats)
    if capped:
        applied_caps = apply_learnable_frobenius_caps(model, cap, norm_dtype=args.cap_norm_dtype)
        model.to(device)
    after_cap_stats = weight_stats(model, args.fast_weight_stats)

    trainable_params_count = configure_trainable(model, args.train_scope, capped=capped)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = make_optimizer(torch, args, trainable_params)
    checkpoints = sorted(set(max(0, step) for step in args.checkpoints))
    max_steps = max(checkpoints)
    warmup_steps = int(args.warmup_ratio * max_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max_steps) if max_steps > 0 else None

    rows = []
    losses = []
    start = time.time()
    variant_name = "baseline" if cap is None else f"fnlearn:{cap:g}"
    kind = "baseline" if cap is None else "fnlearn"

    def evaluate_checkpoint(step):
        metrics = evaluate(torch, model, eval_loader, device)
        value = metrics.get(args.metric)
        caps = collect_learnable_cap_stats(model) if capped else []
        row = {
            "variant": variant_name,
            "kind": kind,
            "cap": cap,
            "checkpoint_steps": step,
            "metric": args.metric,
            "metric_value": value,
            "baseline_target": target,
            "threshold": threshold_value(target, args),
            "meets_threshold": meets_threshold(value, target, args),
            "mean_train_loss": (sum(losses) / len(losses)) if losses else None,
            "runtime_sec": time.time() - start,
            "metrics": metrics,
            "weight_stats_final": weight_stats(model, args.fast_weight_stats),
            "cap_stats": caps,
            "cap_summary": cap_summary(caps),
        }
        rows.append(row)
        print(
            f"{variant_name} step={step} {args.metric}={value} "
            f"threshold={row['threshold']} meets={row['meets_threshold']}",
            flush=True,
        )
        return row["meets_threshold"] is True

    recovered = False
    if 0 in checkpoints:
        recovered = evaluate_checkpoint(0)

    step_count = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    while step_count < max_steps and not (capped and args.early_stop_on_recovery and recovered):
        for batch in train_loader:
            batch = move_batch(batch, device)
            outputs = model(**batch)
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
                recovered = evaluate_checkpoint(step_count)
                model.train()
                if capped and args.early_stop_on_recovery and recovered:
                    break
            if step_count >= max_steps:
                break

    result = {
        "variant": variant_name,
        "kind": kind,
        "cap": cap,
        "applied_caps": applied_caps,
        "trainable_params": trainable_params_count,
        "weight_stats_before": before_stats,
        "weight_stats_after_cap": after_cap_stats,
        "checkpoints": rows,
    }
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def summarize_minimal(results):
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


def main():
    args = parse_args()
    import os
    import torch
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    seed_everything(args.seed)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True
    device = pick_device(torch, args.device)
    config = AutoConfig.from_pretrained(args.model, cache_dir=args.cache_dir, local_files_only=args.offline)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_token_budget = args.max_train_tokens or args.max_eval_tokens or args.seq_len * args.batch_size * max(8, args.grad_accum_steps)
    train_records = chunk_records(tokenizer, load_dataset, args.config, "train", args.seq_len, train_token_budget, args.cache_dir)
    eval_records = chunk_records(tokenizer, load_dataset, args.config, "validation", args.seq_len, args.max_eval_tokens, args.cache_dir)
    train_loader = make_loader(torch, train_records, args.batch_size, shuffle=True)
    eval_loader = make_loader(torch, eval_records, args.eval_batch_size or args.batch_size, shuffle=False)

    target = baseline_target(args)
    baseline_run = None
    if args.include_baseline:
        print(f"running baseline on WikiText with train_scope={args.train_scope}", flush=True)
        baseline_run = train_variant(torch, args, tokenizer, None, None, train_loader, eval_loader, device)
        target = select_baseline_target(baseline_run["checkpoints"], args.metric, args.baseline_selection)

    results = []
    for cap in args.caps:
        print(f"running learnable FN cap {cap:g} on WikiText", flush=True)
        result = train_variant(torch, args, tokenizer, cap, target, train_loader, eval_loader, device)
        results.append(result)
        if args.stop_after_first_recovered and any(row.get("meets_threshold") is True for row in result["checkpoints"]):
            print(f"stopping after first recovered cap {cap:g}", flush=True)
            break

    output = ensure_parent(args.output)
    payload = {
        **now_metadata(),
        "model": args.model,
        "model_config": config_metadata(config),
        "dataset": args.dataset,
        "config": args.config,
        "split": "validation",
        "seed": args.seed,
        "device": str(device),
        "cache_dir": args.cache_dir,
        "offline": args.offline,
        "caps": args.caps,
        "checkpoints": sorted(set(args.checkpoints)),
        "metric": args.metric,
        "baseline_target": target,
        "baseline_selection": args.baseline_selection,
        "relative_tolerance": args.relative_tolerance,
        "absolute_tolerance": args.absolute_tolerance,
        "threshold": threshold_value(target, args),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size or args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "optimizer": args.optimizer,
        "train_scope": args.train_scope,
        "dtype": args.dtype,
        "cap_norm_dtype": args.cap_norm_dtype,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_grad_norm": args.max_grad_norm,
        "early_stop_on_recovery": args.early_stop_on_recovery,
        "stop_after_first_recovered": args.stop_after_first_recovered,
        "fast_weight_stats": args.fast_weight_stats,
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
        "attn_implementation": args.attn_implementation,
        "max_train_tokens": args.max_train_tokens,
        "max_eval_tokens": args.max_eval_tokens,
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
