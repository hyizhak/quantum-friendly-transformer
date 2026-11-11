#!/usr/bin/env python
"""Build the retained SST-2 recovery summary for FN cap versus hidden dimension."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from quantum_friendly_transformer.experiments.lib.retained import RECOVERY_FIELDNAMES, RETAINED_RECOVERY_RUNS


FN_CAP = 24.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("experiments/outputs/raw"),
        help="Directory containing retained raw JSON recovery outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/aggregate_recovery_fn_vs_d/recovery_fn_vs_d_sst2.csv"),
        help="Filtered recovery summary CSV.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def best_checkpoint(checkpoints: list[dict], metric: str) -> dict | None:
    if not checkpoints:
        return None
    higher_is_better = metric not in {"eval_loss", "loss", "perplexity"}
    key = (lambda row: row.get("metric_value", float("-inf"))) if higher_is_better else (
        lambda row: -row.get("metric_value", float("inf"))
    )
    return max(checkpoints, key=key)


def final_checkpoint(checkpoints: list[dict]) -> dict | None:
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda row: row.get("checkpoint_steps", -1))


def first_success(checkpoints: list[dict]) -> dict | None:
    successes = [row for row in checkpoints if row.get("meets_threshold") is True]
    if not successes:
        return None
    return min(successes, key=lambda row: row.get("checkpoint_steps", 0))


def cap_equal(value, cap: float) -> bool:
    if value is None:
        return False
    try:
        return float(value) == float(cap)
    except (TypeError, ValueError):
        return False


def result_cap(data: dict, result: dict):
    return result.get("cap", data.get("cap"))


def result_for_cap(data: dict, results: list[dict], cap: float) -> dict | None:
    for result in results:
        if cap_equal(result_cap(data, result), cap):
            return result
    if len(results) == 1 and cap_equal(data.get("cap"), cap):
        return results[0]
    return None


def recovery_results(data: dict) -> list[dict]:
    if data.get("results") is not None:
        return data.get("results") or []
    if data.get("recovery_run") is not None:
        return [data["recovery_run"]]
    return data.get("recovery_runs") or []


def baseline_checkpoints(raw_dir: Path, spec: dict, data: dict) -> list[dict]:
    baseline_source = spec.get("baseline_source")
    baseline_data = read_json(raw_dir / baseline_source) if baseline_source else data
    baseline_run = baseline_data.get("baseline_run")
    if baseline_run is None:
        return []
    return baseline_run.get("checkpoints") or []


def checkpoint_step(row: dict | None):
    return None if row is None else row.get("checkpoint_steps")


def checkpoint_metric(row: dict | None):
    return None if row is None else row.get("metric_value")


def checkpoint_for_metric(checkpoints: list[dict], target) -> dict | None:
    if target is None:
        return None
    try:
        target_value = float(target)
    except (TypeError, ValueError):
        return None
    matches = []
    for row in checkpoints:
        value = row.get("metric_value")
        if value is None:
            continue
        if abs(float(value) - target_value) <= 1e-12:
            matches.append(row)
    if not matches:
        return None
    return min(matches, key=lambda row: row.get("checkpoint_steps", 0))


def recovery_learning_rate(data: dict):
    return data.get("learning_rate", data.get("recovery_learning_rate"))


def recovery_train_scope(data: dict):
    return data.get("train_scope", data.get("recovery_train_scope"))


def row_for_run(raw_dir: Path, spec: dict) -> dict:
    path = raw_dir / spec["source"]
    data = read_json(path)
    if data.get("model") != spec["model"]:
        raise ValueError(f"{path} has model={data.get('model')!r}, expected {spec['model']!r}")

    results = recovery_results(data)
    baseline_source = spec.get("baseline_source")
    metric = data.get("metric", "accuracy")
    baseline_target = spec.get("baseline_target", data.get("baseline_target"))
    threshold = spec.get("threshold", data.get("threshold"))
    baseline_rows = baseline_checkpoints(raw_dir, spec, data)
    baseline_trace_best = best_checkpoint(baseline_rows, metric)
    baseline_final = final_checkpoint(baseline_rows)
    baseline_best_metric = baseline_target if baseline_target is not None else checkpoint_metric(baseline_trace_best)
    baseline_best = checkpoint_for_metric(baseline_rows, baseline_best_metric) or baseline_trace_best

    max_steps = max(data.get("checkpoints") or [0])
    if data.get("recovery_run"):
        max_steps = max(max_steps, max((row.get("checkpoint_steps", 0) for row in data["recovery_run"].get("checkpoints", [])), default=0))
    if baseline_rows:
        max_steps = max(max_steps, max((row.get("checkpoint_steps", 0) for row in baseline_rows), default=0))

    fn24_result = result_for_cap(data, results, FN_CAP)
    fn24_checkpoints = fn24_result.get("checkpoints") if fn24_result is not None else []
    fn24_checkpoints = fn24_checkpoints or []
    fn24_first = first_success(fn24_checkpoints)
    fn24_best = best_checkpoint(fn24_checkpoints, data.get("metric", "accuracy"))
    fn24_final = final_checkpoint(fn24_checkpoints)

    return {
        "model": spec["model"],
        "family": spec["family"],
        "hidden_size_d": spec["hidden_size_d"],
        "task": data.get("task"),
        "metric": data.get("metric"),
        "tensor_N": data.get("tensor_sequence_length") or data.get("max_length"),
        "protocol": spec["protocol"],
        "train_scope": recovery_train_scope(data),
        "optimizer": data.get("optimizer"),
        "learning_rate": recovery_learning_rate(data),
        "batch_size": data.get("batch_size"),
        "grad_accum_steps": data.get("grad_accum_steps"),
        "max_steps": max_steps,
        "baseline_target": baseline_target,
        "baseline_best_steps": checkpoint_step(baseline_best),
        "baseline_best_metric": baseline_best_metric,
        "baseline_final_steps": checkpoint_step(baseline_final),
        "baseline_final_metric": checkpoint_metric(baseline_final),
        "threshold": threshold,
        "fn_cap": f"{FN_CAP:g}",
        "fn24_recovered": fn24_first is not None,
        "fn24_first_recovery_steps": None if fn24_first is None else fn24_first.get("checkpoint_steps"),
        "fn24_first_recovery_metric": None if fn24_first is None else fn24_first.get("metric_value"),
        "fn24_best_steps": None if fn24_best is None else fn24_best.get("checkpoint_steps"),
        "fn24_best_metric": None if fn24_best is None else fn24_best.get("metric_value"),
        "fn24_final_steps": None if fn24_final is None else fn24_final.get("checkpoint_steps"),
        "fn24_final_metric": None if fn24_final is None else fn24_final.get("metric_value"),
        "fn24_cap_over_d": FN_CAP / spec["hidden_size_d"],
        "source_file": str(path),
        "baseline_source_file": "" if baseline_source is None else str(raw_dir / baseline_source),
    }


def main() -> None:
    args = parse_args()
    rows = [row_for_run(args.raw_dir, spec) for spec in RETAINED_RECOVERY_RUNS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECOVERY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
