#!/usr/bin/env python
"""Aggregate retained FN24 plus trainable-attention SST-2 recoveries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from quantum_friendly_transformer.experiments.lib.retained import RETAINED_TRAINABLE_ATTENTION_RECOVERY_RUNS
from quantum_friendly_transformer.experiments.lib.checkpoints import recovery_checkpoint_path


FIELDNAMES = [
    "model",
    "task",
    "cap",
    "baseline_target",
    "threshold",
    "first_recovery_steps",
    "first_recovery_accuracy",
    "best_accuracy",
    "applied_caps",
    "attention_layers",
    "scale_layout",
    "scale_controllers",
    "initial_score_scale_mean",
    "final_score_scale_mean",
    "final_score_scale_min",
    "final_score_scale_max",
    "native_score_scale_mean",
    "mean_attention_scale_grad_norm",
    "scale_frozen_after_warmup",
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
        default=Path("experiments/outputs/aggregate_trainable_attention_recovery"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("results/sst2_fn24_trainable_attention_recovery.md"),
    )
    return parser.parse_args()


def first_success(checkpoints: list[dict]) -> dict | None:
    successes = [row for row in checkpoints if row.get("meets_threshold") is True]
    return min(successes, key=lambda row: row["checkpoint_steps"]) if successes else None


def native_scales(result: dict) -> list[float]:
    values = [
        float(row["native_score_scale"])
        for row in result.get("attention_scaling_layers", [])
        if row.get("native_score_scale") is not None
    ]
    if values:
        return values
    backends = {
        row.get("backend") for row in result.get("attention_scaling_layers", [])
    }
    return [1.0] if backends == {"gpt_neo"} else []


def build_row(output_root: Path, retained: dict) -> dict:
    source = output_root / retained["source"]
    if not source.exists():
        raise FileNotFoundError(source)
    data = json.loads(source.read_text())
    if data.get("model") != retained["model"]:
        raise ValueError(f"model mismatch in {source}: {data.get('model')!r}")
    if data.get("task") != "sst2" or not data.get("trainable_attention_scaling"):
        raise ValueError(f"not a trainable-attention SST-2 artifact: {source}")

    cap_results = [row for row in data.get("results", []) if row.get("cap") == 24.0]
    if len(cap_results) != 1:
        raise ValueError(f"expected exactly one FN24 result in {source}")
    result = cap_results[0]
    checkpoints = result.get("checkpoints") or []
    first = first_success(checkpoints)
    if first is None or not result.get("recovered"):
        raise ValueError(f"retained result is not recovered: {source}")

    scale_stats = result.get("attention_scale_stats_final") or []
    if not scale_stats:
        raise ValueError(f"missing final attention-scale stats: {source}")
    initial_scales = [float(row["initial_score_scale"]) for row in scale_stats]
    final_scales = [float(row["score_scale"]) for row in scale_stats]
    native = native_scales(result)
    frozen = bool(result.get("attention_scale_frozen_after_warmup", False))
    scale_grad = first.get("mean_attention_scale_grad_norm")
    changed = any(
        not math.isclose(initial, final, rel_tol=1e-9, abs_tol=0.0)
        for initial, final in zip(initial_scales, final_scales, strict=True)
    )
    warmup = result.get("attention_scale_warmup")
    task_trainable_evidence = (
        not frozen
        and (
            (scale_grad is not None and float(scale_grad) > 0.0)
            or (warmup is None and changed)
        )
    )
    if not task_trainable_evidence:
        raise ValueError(f"no task-trainable scale evidence in {source}")

    checkpoint = recovery_checkpoint_path(source, result.get("recovery_checkpoint") or "")
    best = max(float(row["metric_value"]) for row in checkpoints)
    shared = bool(data.get("shared_attention_scale"))
    return {
        "model": data["model"],
        "task": data["task"],
        "cap": 24.0,
        "baseline_target": data["baseline_target"],
        "threshold": data["threshold"],
        "first_recovery_steps": first["checkpoint_steps"],
        "first_recovery_accuracy": first["metric_value"],
        "best_accuracy": best,
        "applied_caps": result.get("applied_caps"),
        "attention_layers": len(result.get("attention_scaling_layers") or []),
        "scale_layout": "shared" if shared else "per_layer",
        "scale_controllers": len(scale_stats),
        "initial_score_scale_mean": mean(initial_scales),
        "final_score_scale_mean": mean(final_scales),
        "final_score_scale_min": min(final_scales),
        "final_score_scale_max": max(final_scales),
        "native_score_scale_mean": mean(native) if native else None,
        "mean_attention_scale_grad_norm": scale_grad,
        "scale_frozen_after_warmup": frozen,
        "task_trainable_scale_evidence": task_trainable_evidence,
        "recovery_checkpoint": str(checkpoint),
        "recovery_checkpoint_bytes": checkpoint.stat().st_size,
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
        "cap": 24.0,
        "attention_scaling": "trainable_alpha_s_alpha_w",
        "expected_models": len(RETAINED_TRAINABLE_ATTENTION_RECOVERY_RUNS),
        "recovered_models": len(rows),
        "complete": len(rows) == len(RETAINED_TRAINABLE_ATTENTION_RECOVERY_RUNS),
        "results": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def display_model(model: str) -> str:
    return model.split("/", 1)[-1]


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# SST-2 FN24 recovery with trainable alpha attention scaling",
        "",
        "All nine retained model variants recovered with a learnable Frobenius cap of 24 and a score-scale controller initialized from `1 / (alpha_s^2 * alpha_w^2)`. Recovery means validation accuracy is within 0.01 of the corresponding uncapped target.",
        "",
        "Models are loaded through Hugging Face `transformers.AutoModel*`. The runner imports both `apply_learnable_frobenius_caps` and `apply_attention_scaling` from `quantum_friendly_transformer.norm_transformer`; there is no second Hugging Face-specific FN implementation.",
        "",
        "The 33M, 2Layers-33M, OLMo, and Mistral controllers received a controller-only native prior warmup and then remained trainable during SST-2. Earlier TinyStories variants trained their controllers directly from the calibrated alpha initialization. Every retained row has either a recorded nonzero task gradient norm or direct task-phase scale movement, and none freezes the controller.",
        "",
        "| Model | Scale layout | Threshold | First recovery step | Recovery accuracy | Final scale mean | Task-trainable evidence |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| `{model}` | {layout} | {threshold:.6f} | {step:,} | {accuracy:.6f} | {scale:.6g} | yes |".format(
                model=display_model(row["model"]),
                layout=row["scale_layout"],
                threshold=row["threshold"],
                step=row["first_recovery_steps"],
                accuracy=row["first_recovery_accuracy"],
                scale=row["final_score_scale_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Raw artifacts",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- `{row['source_file']}`")
        lines.append(f"  - checkpoint: `{row['recovery_checkpoint']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = [
        build_row(args.output_root, retained)
        for retained in RETAINED_TRAINABLE_ATTENTION_RECOVERY_RUNS
    ]
    write_csv(args.output_dir / "summary.csv", rows)
    write_json(args.output_dir / "summary.json", rows)
    write_markdown(args.markdown, rows)
    print(
        f"aggregated {len(rows)}/{len(RETAINED_TRAINABLE_ATTENTION_RECOVERY_RUNS)} "
        f"recovered models into {args.output_dir}"
    )


if __name__ == "__main__":
    main()
