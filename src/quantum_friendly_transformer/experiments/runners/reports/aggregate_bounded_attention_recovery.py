#!/usr/bin/env python
"""Aggregate strict alpha-denominator-bound SST-2 recovery reruns."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from quantum_friendly_transformer.experiments.lib.retained import RETAINED_BOUNDED_ATTENTION_RECOVERY_RUNS


FN_CAP = 24.0
FIELDNAMES = [
    "model",
    "task",
    "cap",
    "baseline_target",
    "threshold",
    "recovered",
    "first_recovery_steps",
    "first_recovery_accuracy",
    "best_accuracy",
    "final_steps",
    "final_accuracy",
    "applied_caps",
    "attention_layers",
    "scale_layout",
    "scale_controllers",
    "configured_alpha_s",
    "effective_alpha_s",
    "alpha_w",
    "theoretical_denominator_floor",
    "implemented_denominator_floor_min",
    "implemented_denominator_floor_max",
    "initial_score_scale_mean",
    "final_score_scale_mean",
    "final_score_scale_min",
    "final_score_scale_max",
    "final_score_denominator_min",
    "final_score_denominator_mean",
    "final_score_denominator_max",
    "final_scale_to_alpha_ratio_max",
    "max_observed_scale_to_alpha_ratio",
    "min_observed_denominator_to_alpha_ratio",
    "adapter_rows_checked",
    "scale_stat_rows_checked",
    "all_observed_constraints_satisfied",
    "mean_attention_scale_grad_norm",
    "task_trainable_scale_evidence",
    "recovery_checkpoint",
    "recovery_checkpoint_bytes",
    "source_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/outputs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/aggregate_bounded_attention_recovery"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("results/sst2_fn24_bounded_attention_recovery.md"),
    )
    return parser.parse_args()


def _positive_finite(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return number


def _finite(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _validate_common_scale_values(
    row: dict,
    *,
    context: str,
    theoretical_floor: float,
) -> dict:
    scale = _positive_finite(row.get("score_scale"), f"{context}.score_scale")
    denominator = _positive_finite(
        row.get("score_denominator"),
        f"{context}.score_denominator",
    )
    ceiling = _positive_finite(
        row.get("score_scale_ceiling"),
        f"{context}.score_scale_ceiling",
    )
    floor = _positive_finite(
        row.get("score_denominator_floor"),
        f"{context}.score_denominator_floor",
    )
    if scale > ceiling:
        raise ValueError(f"{context} violates score_scale <= score_scale_ceiling")
    if denominator < floor:
        raise ValueError(
            f"{context} violates score_denominator >= score_denominator_floor"
        )
    if floor < theoretical_floor:
        raise ValueError(
            f"{context}.score_denominator_floor={floor!r} is below "
            f"effective_alpha_s**2 * alpha_w**2={theoretical_floor!r}"
        )
    _require_close(denominator, 1.0 / scale, f"{context}.score_denominator")
    _require_close(floor, 1.0 / ceiling, f"{context}.score_denominator_floor")
    return {
        "scale": scale,
        "denominator": denominator,
        "ceiling": ceiling,
        "floor": floor,
        "scale_to_alpha_ratio": scale / ceiling,
        "denominator_to_alpha_ratio": denominator / floor,
    }


def validate_adapter_row(
    row: dict,
    *,
    context: str,
    effective_alpha_s: float,
    alpha_w: float,
    theoretical_floor: float,
) -> dict:
    if row.get("scaling") != "alpha_s_alpha_w":
        raise ValueError(f"{context} is not alpha_s_alpha_w scaling")
    if row.get("trainable") is not True:
        raise ValueError(f"{context} is not trainable")
    if row.get("enforce_alpha_denominator_bound") is not True:
        raise ValueError(f"{context} does not enforce the alpha denominator bound")
    _require_close(
        _positive_finite(row.get("alpha_s"), f"{context}.alpha_s"),
        effective_alpha_s,
        f"{context}.alpha_s",
    )
    _require_close(
        _positive_finite(row.get("alpha_w"), f"{context}.alpha_w"),
        alpha_w,
        f"{context}.alpha_w",
    )
    return _validate_common_scale_values(
        row,
        context=context,
        theoretical_floor=theoretical_floor,
    )


def validate_scale_stat(
    row: dict,
    *,
    context: str,
    theoretical_floor: float,
) -> dict:
    if row.get("controller") != "AlphaBoundedAttentionScale":
        raise ValueError(f"{context} does not use AlphaBoundedAttentionScale")
    if row.get("alpha_denominator_bound_enforced") is not True:
        raise ValueError(f"{context} does not enforce the alpha denominator bound")
    values = _validate_common_scale_values(
        row,
        context=context,
        theoretical_floor=theoretical_floor,
    )
    ratio = _positive_finite(
        row.get("scale_to_alpha_ratio"),
        f"{context}.scale_to_alpha_ratio",
    )
    if ratio > 1.0:
        raise ValueError(f"{context} violates scale_to_alpha_ratio <= 1")
    denominator_ratio = _positive_finite(
        row.get("denominator_to_alpha_ratio"),
        f"{context}.denominator_to_alpha_ratio",
    )
    if denominator_ratio < 1.0:
        raise ValueError(
            f"{context} violates denominator_to_alpha_ratio >= 1"
        )
    _require_close(
        ratio,
        values["scale_to_alpha_ratio"],
        f"{context}.scale_to_alpha_ratio",
    )
    _require_close(
        denominator_ratio,
        values["denominator_to_alpha_ratio"],
        f"{context}.denominator_to_alpha_ratio",
    )
    slack = _finite(
        row.get("log_denominator_slack"),
        f"{context}.log_denominator_slack",
    )
    if slack < 0.0:
        raise ValueError(f"{context} violates log_denominator_slack >= 0")
    if row.get("raw_scale") is not None:
        raise ValueError(f"{context}.raw_scale must be null for the bounded controller")
    initial = _positive_finite(
        row.get("initial_score_scale"),
        f"{context}.initial_score_scale",
    )
    maximum = _positive_finite(
        row.get("max_score_scale"),
        f"{context}.max_score_scale",
    )
    _require_close(initial, values["ceiling"], f"{context}.initial_score_scale")
    _require_close(maximum, values["ceiling"], f"{context}.max_score_scale")
    values["scale_to_alpha_ratio"] = ratio
    values["denominator_to_alpha_ratio"] = denominator_ratio
    return values


def first_success(checkpoints: list[dict]) -> dict | None:
    successes = [row for row in checkpoints if row.get("meets_threshold") is True]
    if not successes:
        return None
    return min(successes, key=lambda row: row["checkpoint_steps"])


def _validate_checkpoint_path(source: Path, value: str) -> Path:
    from quantum_friendly_transformer.experiments.lib.checkpoints import recovery_checkpoint_path

    return recovery_checkpoint_path(source, value)


def build_row(output_root: Path, retained: dict) -> dict:
    source = output_root / retained["source"]
    if not source.exists():
        raise FileNotFoundError(source)
    data = json.loads(source.read_text())
    if data.get("model") != retained["model"]:
        raise ValueError(f"model mismatch in {source}: {data.get('model')!r}")
    if data.get("task") != "sst2":
        raise ValueError(f"not an SST-2 artifact: {source}")
    if data.get("attention_scaling") != "alpha_s_alpha_w":
        raise ValueError(f"not an alpha_s_alpha_w artifact: {source}")
    if data.get("trainable_attention_scaling") is not True:
        raise ValueError(f"not a trainable-attention artifact: {source}")
    if data.get("enforce_alpha_denominator_bound") is not True:
        raise ValueError(f"not a strict alpha-denominator-bound artifact: {source}")
    if float(data.get("attention_scale_native_regularization", 0.0)) != 0.0:
        raise ValueError(f"bounded artifact uses native-scale regularization: {source}")
    if int(data.get("attention_scale_prior_warmup_steps", 0)) != 0:
        raise ValueError(f"bounded artifact uses native-prior warmup: {source}")
    if data.get("attention_scale_prior_warmup_snap_native", False) is not False:
        raise ValueError(f"bounded artifact snaps its controller to native: {source}")

    cap_results = [row for row in data.get("results", []) if row.get("cap") == FN_CAP]
    if len(cap_results) != 1:
        raise ValueError(f"expected exactly one FN24 result in {source}")
    result = cap_results[0]
    if result.get("alpha_denominator_bound_enforced") is not True:
        raise ValueError(f"FN24 result does not record the strict bound: {source}")
    if result.get("attention_scale_frozen_after_warmup", False) is not False:
        raise ValueError(f"bounded controller was frozen in {source}")

    configured_alpha_s = _positive_finite(data.get("alpha_s"), f"{source}.alpha_s")
    effective_alpha_s = _positive_finite(
        result.get("effective_alpha_s"),
        f"{source}.effective_alpha_s",
    )
    if effective_alpha_s < configured_alpha_s:
        raise ValueError(f"effective_alpha_s is below configured alpha_s in {source}")
    alpha_w = _positive_finite(data.get("alpha_w"), f"{source}.alpha_w")
    theoretical_floor = effective_alpha_s**2 * alpha_w**2
    if not math.isfinite(theoretical_floor):
        raise ValueError(f"non-finite theoretical denominator floor in {source}")

    adapter_rows = result.get("attention_scaling_layers") or []
    if not adapter_rows:
        raise ValueError(f"missing attention-scaling adapter rows: {source}")
    adapter_observations = [
        validate_adapter_row(
            row,
            context=f"{source}.attention_scaling_layers[{index}]",
            effective_alpha_s=effective_alpha_s,
            alpha_w=alpha_w,
            theoretical_floor=theoretical_floor,
        )
        for index, row in enumerate(adapter_rows)
    ]

    checkpoints = result.get("checkpoints") or []
    if not checkpoints:
        raise ValueError(f"missing evaluation checkpoints: {source}")
    zero_checkpoints = [row for row in checkpoints if row.get("checkpoint_steps") == 0]
    if len(zero_checkpoints) != 1:
        raise ValueError(f"expected exactly one step-zero checkpoint in {source}")

    scale_observations = []
    expected_controller_names = None
    initial_stats = None
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        step = checkpoint.get("checkpoint_steps")
        stats = checkpoint.get("attention_scale_stats") or []
        if not stats:
            raise ValueError(f"missing attention-scale stats at step {step} in {source}")
        names = [row.get("name") for row in stats]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate attention-scale controller names at step {step}")
        if expected_controller_names is None:
            expected_controller_names = set(names)
        elif set(names) != expected_controller_names:
            raise ValueError(f"attention-scale controller set changed at step {step}")
        if step == 0:
            initial_stats = stats
        for stat_index, stat in enumerate(stats):
            scale_observations.append(
                validate_scale_stat(
                    stat,
                    context=(
                        f"{source}.checkpoints[{checkpoint_index}]"
                        f".attention_scale_stats[{stat_index}]"
                    ),
                    theoretical_floor=theoretical_floor,
                )
            )

    final_stats = result.get("attention_scale_stats_final") or []
    if not final_stats:
        raise ValueError(f"missing final attention-scale stats: {source}")
    if {row.get("name") for row in final_stats} != expected_controller_names:
        raise ValueError(f"final attention-scale controller set changed in {source}")
    final_observations = [
        validate_scale_stat(
            stat,
            context=f"{source}.attention_scale_stats_final[{index}]",
            theoretical_floor=theoretical_floor,
        )
        for index, stat in enumerate(final_stats)
    ]
    scale_observations.extend(final_observations)

    expected_shared = bool(data.get("shared_attention_scale", False))
    if any(bool(row.get("shared")) != expected_shared for row in final_stats):
        raise ValueError(f"scale layout metadata mismatch in {source}")
    if expected_shared and len(final_stats) != 1:
        raise ValueError(f"shared scaling must have exactly one controller in {source}")

    first = first_success(checkpoints)
    recovered = first is not None
    if result.get("recovered") is not recovered:
        raise ValueError(f"recovery status disagrees with checkpoints in {source}")
    checkpoint_value = result.get("recovery_checkpoint")
    recovery_checkpoint = None
    recovery_checkpoint_bytes = None
    if recovered:
        if not checkpoint_value:
            raise ValueError(f"recovered result has no recovery checkpoint: {source}")
        recovery_checkpoint = _validate_checkpoint_path(source, checkpoint_value)
        recovery_checkpoint_bytes = recovery_checkpoint.stat().st_size
    elif checkpoint_value:
        raise ValueError(f"non-recovered result reports a recovery checkpoint: {source}")

    final_checkpoint = max(checkpoints, key=lambda row: row["checkpoint_steps"])
    best_accuracy = max(float(row["metric_value"]) for row in checkpoints)
    initial_scales = [float(row["initial_score_scale"]) for row in initial_stats]
    final_scales = [item["scale"] for item in final_observations]
    final_denominators = [item["denominator"] for item in final_observations]
    final_floors = [item["floor"] for item in final_observations]
    changed = any(
        not math.isclose(initial, final, rel_tol=1e-9, abs_tol=0.0)
        for initial, final in zip(initial_scales, final_scales, strict=True)
    )
    scale_grad = next(
        (
            float(row["mean_attention_scale_grad_norm"])
            for row in reversed(checkpoints)
            if row.get("mean_attention_scale_grad_norm") is not None
        ),
        None,
    )
    task_trainable_evidence = bool(
        changed or (scale_grad is not None and scale_grad > 0.0)
    )
    all_observations = adapter_observations + scale_observations
    return {
        "model": data["model"],
        "task": data["task"],
        "cap": FN_CAP,
        "baseline_target": data["baseline_target"],
        "threshold": data["threshold"],
        "recovered": recovered,
        "first_recovery_steps": None if first is None else first["checkpoint_steps"],
        "first_recovery_accuracy": None if first is None else first["metric_value"],
        "best_accuracy": best_accuracy,
        "final_steps": final_checkpoint["checkpoint_steps"],
        "final_accuracy": final_checkpoint["metric_value"],
        "applied_caps": result.get("applied_caps"),
        "attention_layers": len(adapter_rows),
        "scale_layout": "shared" if expected_shared else "per_layer",
        "scale_controllers": len(final_stats),
        "configured_alpha_s": configured_alpha_s,
        "effective_alpha_s": effective_alpha_s,
        "alpha_w": alpha_w,
        "theoretical_denominator_floor": theoretical_floor,
        "implemented_denominator_floor_min": min(final_floors),
        "implemented_denominator_floor_max": max(final_floors),
        "initial_score_scale_mean": mean(initial_scales),
        "final_score_scale_mean": mean(final_scales),
        "final_score_scale_min": min(final_scales),
        "final_score_scale_max": max(final_scales),
        "final_score_denominator_min": min(final_denominators),
        "final_score_denominator_mean": mean(final_denominators),
        "final_score_denominator_max": max(final_denominators),
        "final_scale_to_alpha_ratio_max": max(
            item["scale_to_alpha_ratio"] for item in final_observations
        ),
        "max_observed_scale_to_alpha_ratio": max(
            item["scale_to_alpha_ratio"] for item in all_observations
        ),
        "min_observed_denominator_to_alpha_ratio": min(
            item["denominator_to_alpha_ratio"] for item in all_observations
        ),
        "adapter_rows_checked": len(adapter_observations),
        "scale_stat_rows_checked": len(scale_observations),
        "all_observed_constraints_satisfied": True,
        "mean_attention_scale_grad_norm": scale_grad,
        "task_trainable_scale_evidence": task_trainable_evidence,
        "recovery_checkpoint": (
            str(recovery_checkpoint) if recovery_checkpoint is not None else None
        ),
        "recovery_checkpoint_bytes": recovery_checkpoint_bytes,
        "source_file": str(source),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict]) -> None:
    payload = {
        "task": "sst2",
        "cap": FN_CAP,
        "attention_scaling": "trainable_alpha_s_alpha_w",
        "alpha_denominator_bound_enforced": True,
        "constraint": {
            "score_scale": "s(t) <= 1 / (effective_alpha_s^2 * alpha_w^2)",
            "score_denominator": "D(t) >= effective_alpha_s^2 * alpha_w^2",
        },
        "expected_models": len(RETAINED_BOUNDED_ATTENTION_RECOVERY_RUNS),
        "completed_models": len(rows),
        "recovered_models": sum(bool(row["recovered"]) for row in rows),
        "complete": len(rows) == len(RETAINED_BOUNDED_ATTENTION_RECOVERY_RUNS),
        "all_observed_constraints_satisfied": all(
            row["all_observed_constraints_satisfied"] for row in rows
        ),
        "results": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def display_model(model: str) -> str:
    return model.split("/", 1)[-1]


def write_markdown(path: Path, rows: list[dict]) -> None:
    recovered = sum(bool(row["recovered"]) for row in rows)
    checked = sum(int(row["scale_stat_rows_checked"]) for row in rows)
    lines = [
        "# SST-2 FN24 recovery with a strict alpha denominator bound",
        "",
        f"All {len(rows)} bounded reruns completed; {recovered}/{len(rows)} met their corresponding SST-2 recovery thresholds. A failed recovery remains a valid completed strict-bound experiment.",
        "",
        "Every adapter-install row, every recorded evaluation checkpoint, and every final controller row satisfies `scale_to_alpha_ratio <= 1` and `score_denominator >= score_denominator_floor`. The report checked {checked:,} recorded controller rows. The implemented floor is also no smaller than `effective_alpha_s^2 * alpha_w^2`, including calibrated alpha values.".format(checked=checked),
        "",
        "| Model | Layout | Threshold | Recovered | First recovery | Best accuracy | Final scale / alpha max | Minimum D / floor |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        first_step = (
            "—"
            if row["first_recovery_steps"] is None
            else f"{int(row['first_recovery_steps']):,}"
        )
        lines.append(
            "| `{model}` | {layout} | {threshold:.6f} | {recovered} | {step} | "
            "{best:.6f} | {scale_ratio:.9g} | {denominator_ratio:.9g} |".format(
                model=display_model(row["model"]),
                layout=row["scale_layout"],
                threshold=row["threshold"],
                recovered="yes" if row["recovered"] else "no",
                step=first_step,
                best=row["best_accuracy"],
                scale_ratio=row["max_observed_scale_to_alpha_ratio"],
                denominator_ratio=row[
                    "min_observed_denominator_to_alpha_ratio"
                ],
            )
        )
    lines.extend(["", "## Raw artifacts", ""])
    for row in rows:
        lines.append(f"- `{row['source_file']}`")
        if row["recovery_checkpoint"]:
            lines.append(f"  - recovery checkpoint: `{row['recovery_checkpoint']}`")
        else:
            lines.append("  - recovery checkpoint: none (threshold not met)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = [
        build_row(args.output_root, retained)
        for retained in RETAINED_BOUNDED_ATTENTION_RECOVERY_RUNS
    ]
    write_csv(args.output_dir / "summary.csv", rows)
    write_json(args.output_dir / "summary.json", rows)
    write_markdown(args.markdown, rows)
    recovered = sum(bool(row["recovered"]) for row in rows)
    print(
        f"aggregated {len(rows)}/{len(RETAINED_BOUNDED_ATTENTION_RECOVERY_RUNS)} "
        f"completed strict-bound models ({recovered} recovered) into {args.output_dir}"
    )


if __name__ == "__main__":
    main()
