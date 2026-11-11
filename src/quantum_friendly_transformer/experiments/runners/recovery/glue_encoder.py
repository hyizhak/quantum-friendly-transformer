import argparse
import time

from quantum_friendly_transformer.norm_transformer.frobenius_cap import (
    apply_learnable_frobenius_caps,
    collect_learnable_cap_stats,
    collect_weight_stats,
)

from quantum_friendly_transformer.experiments.lib.common import (
    ensure_parent,
    glue_num_labels,
    glue_split_name,
    metadata_path,
    now_metadata,
    package_versions,
    seed_everything,
    write_json,
)
from quantum_friendly_transformer.experiments.runners.normalisation.glue import (
    GLUE_TEXT_COLUMNS,
    evaluate,
    make_loaders,
    move_batch,
    pick_device,
    prepare_dataset,
)


HIGHER_IS_BETTER = {
    "accuracy": True,
    "f1": True,
    "matthews": True,
    "pearson": True,
    "spearman": True,
    "eval_loss": False,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Find recovery checkpoints for GLUE learnable-cap FN models.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=sorted(GLUE_TEXT_COLUMNS))
    parser.add_argument("--caps", type=float, nargs="+", required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--baseline-target", type=float, default=None)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def primary_metric(task):
    if task in {"mrpc", "qqp"}:
        return "f1"
    if task == "cola":
        return "matthews"
    if task == "stsb":
        return "pearson"
    return "accuracy"


def baseline_target(args, metric):
    if args.baseline_target is not None:
        return args.baseline_target
    if not args.baseline_json:
        return None
    import json
    from pathlib import Path

    payload = json.loads(Path(args.baseline_json).read_text())
    candidates = [row for row in payload.get("results", []) if row.get("variant") == "baseline"]
    if not candidates:
        raise ValueError(f"no baseline result found in {args.baseline_json}")
    value = candidates[0].get("metrics", {}).get(metric)
    if value is None:
        raise ValueError(f"metric {metric!r} missing from baseline in {args.baseline_json}")
    return float(value)


def threshold_value(target, metric, tolerance):
    if target is None:
        return None
    if HIGHER_IS_BETTER.get(metric, True):
        return target - tolerance
    return target + tolerance


def meets_threshold(value, target, metric, tolerance):
    if target is None or value is None:
        return None
    threshold = threshold_value(target, metric, tolerance)
    if HIGHER_IS_BETTER.get(metric, True):
        return value >= threshold
    return value <= threshold


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


def train_cap(torch, args, cap, metric, target, train_loader, eval_loader, device):
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    seed_everything(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=glue_num_labels(args.task),
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    if args.task == "stsb":
        model.config.problem_type = "regression"
    model.to(device)
    before_stats = collect_weight_stats(model)
    applied_caps = apply_learnable_frobenius_caps(model, cap)
    model.to(device)
    after_cap_stats = collect_weight_stats(model)

    checkpoints = sorted(set(max(0, step) for step in args.checkpoints))
    max_steps = max(checkpoints)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_steps = int(args.warmup_ratio * max_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max_steps) if max_steps > 0 else None

    rows = []
    losses = []
    start = time.time()

    def evaluate_checkpoint(step):
        metrics = evaluate(torch, model, eval_loader, device, args.task)
        value = metrics.get(metric)
        caps = collect_learnable_cap_stats(model)
        rows.append(
            {
                "variant": f"fnlearn:{cap:g}",
                "kind": "fnlearn",
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
                "weight_stats_final": collect_weight_stats(model),
                "cap_stats": caps,
                "cap_summary": cap_summary(caps),
            }
        )

    if 0 in checkpoints:
        evaluate_checkpoint(0)

    step_count = 0
    model.train()
    while step_count < max_steps:
        for batch in train_loader:
            batch = move_batch(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu().item()))
            step_count += 1
            if step_count in checkpoints:
                evaluate_checkpoint(step_count)
                model.train()
            if step_count >= max_steps:
                break

    return {
        "cap": cap,
        "applied_caps": applied_caps,
        "weight_stats_before": before_stats,
        "weight_stats_after_cap": after_cap_stats,
        "checkpoints": rows,
    }


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
    from transformers import AutoTokenizer

    seed_everything(args.seed)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    device = pick_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    train_ds, eval_ds = prepare_dataset(args, tokenizer, load_dataset)
    train_loader, eval_loader = make_loaders(torch, args, tokenizer, train_ds, eval_ds)

    metric = args.metric or primary_metric(args.task)
    target = baseline_target(args, metric)
    results = []
    for cap in args.caps:
        print(f"running learnable FN cap {cap:g} on {args.task}", flush=True)
        results.append(train_cap(torch, args, cap, metric, target, train_loader, eval_loader, device))

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
        "caps": args.caps,
        "checkpoints": sorted(set(args.checkpoints)),
        "metric": metric,
        "baseline_target": target,
        "tolerance": args.tolerance,
        "threshold": threshold_value(target, metric, args.tolerance),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "package_versions": package_versions(),
        "results": results,
        "minimal_recovery": summarize_minimal(results),
    }
    write_json(output, payload)
    write_json(metadata_path(output), {k: v for k, v in payload.items() if k != "results"})
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
