#!/usr/bin/env python
"""Aggregate learnable Frobenius-cap recovery sweeps."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path


FIELDNAMES = [
    "source_file",
    "model",
    "model_type",
    "hidden_dim",
    "num_layers",
    "task",
    "dataset",
    "metric",
    "baseline_target",
    "threshold",
    "cap",
    "recovered",
    "minimal_recovery_steps",
    "minimal_metric_value",
    "max_checkpoint_steps",
    "final_metric_value",
    "final_alpha_w",
    "final_gamma_max",
    "final_gamma_mean",
    "seq_len",
    "train_scope",
    "optimizer",
    "dtype",
    "learning_rate",
    "grad_accum_steps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments/outputs/raw"),
        help="Directory containing recovery_search_learncap_true_*.json files.",
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        help="Optional glob for learnable-cap recovery JSON files. Overrides --input-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/aggregate_learncap_recovery/minimal_recovery_summary.csv"),
        help="Cap-by-cap CSV output path.",
    )
    parser.add_argument(
        "--minimal-output",
        type=Path,
        default=None,
        help="Optional one-row-per-run CSV containing only the smallest recovered cap.",
    )
    return parser.parse_args()


def input_paths(args: argparse.Namespace) -> list[Path]:
    if args.input_glob:
        return [Path(path) for path in sorted(glob.glob(args.input_glob)) if not path.endswith(".metadata.json")]
    return [path for path in sorted(args.input_dir.glob("recovery_search_learncap_true_*.json")) if not path.name.endswith(".metadata.json")]


def first_success(checkpoints: list[dict]) -> dict | None:
    successes = [row for row in checkpoints if row.get("meets_threshold") is True]
    if not successes:
        return None
    return min(successes, key=lambda row: row["checkpoint_steps"])


def max_weight_norm(row: dict) -> float | None:
    stats = row.get("weight_stats_final") or []
    values = [item["frobenius_norm"] for item in stats if item.get("frobenius_norm") is not None]
    return max(values) if values else None


def build_rows(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    for path in input_paths(args):
        data = json.loads(path.read_text())
        task = data.get("task") or data.get("config") or data.get("dataset")
        model_config = data.get("model_config") or {}
        for result in data.get("results", []):
            checkpoints = result.get("checkpoints") or []
            if not checkpoints:
                continue
            final = max(checkpoints, key=lambda row: row["checkpoint_steps"])
            first = first_success(checkpoints)
            cap_summary = final.get("cap_summary", {})
            rows.append(
                {
                    "source_file": str(path),
                    "model": data.get("model"),
                    "model_type": model_config.get("model_type"),
                    "hidden_dim": model_config.get("hidden_dim"),
                    "num_layers": model_config.get("num_layers"),
                    "task": task,
                    "dataset": data.get("dataset"),
                    "metric": data.get("metric"),
                    "baseline_target": data.get("baseline_target"),
                    "threshold": data.get("threshold"),
                    "cap": result.get("cap"),
                    "recovered": first is not None,
                    "minimal_recovery_steps": "" if first is None else first["checkpoint_steps"],
                    "minimal_metric_value": "" if first is None else first["metric_value"],
                    "max_checkpoint_steps": final["checkpoint_steps"],
                    "final_metric_value": final.get("metric_value"),
                    "final_alpha_w": max_weight_norm(final),
                    "final_gamma_max": cap_summary.get("gamma_max"),
                    "final_gamma_mean": cap_summary.get("gamma_mean"),
                    "seq_len": data.get("seq_len"),
                    "train_scope": data.get("train_scope"),
                    "optimizer": data.get("optimizer"),
                    "dtype": data.get("dtype"),
                    "learning_rate": data.get("learning_rate"),
                    "grad_accum_steps": data.get("grad_accum_steps"),
                }
            )
    return rows


def minimal_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["source_file"], row["model"], row["hidden_dim"], row["task"], row["metric"])
        grouped.setdefault(key, []).append(row)

    out = []
    for group_rows in grouped.values():
        recovered = [row for row in group_rows if row["recovered"]]
        if not recovered:
            continue
        out.append(min(recovered, key=lambda row: (float(row["cap"]), int(row["minimal_recovery_steps"]))))
    return out


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    source = args.input_glob or str(args.input_dir)
    if not rows:
        raise SystemExit(f"no learnable-cap recovery files found in {source}")

    write_rows(args.output, rows)
    print(f"wrote {args.output} rows={len(rows)}")

    if args.minimal_output is not None:
        minimal = minimal_rows(rows)
        write_rows(args.minimal_output, minimal)
        print(f"wrote {args.minimal_output} rows={len(minimal)}")


if __name__ == "__main__":
    main()
