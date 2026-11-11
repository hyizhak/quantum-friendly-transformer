#!/usr/bin/env python
"""Run learnable Frobenius-cap recovery sweeps across the OPT family."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


OPT_HIDDEN_DIMS = {
    "facebook/opt-125m": 768,
    "facebook/opt-350m": 1024,
    "facebook/opt-1.3b": 2048,
    "facebook/opt-2.7b": 2560,
    "facebook/opt-6.7b": 4096,
    "facebook/opt-30b": 7168,
}

DEFAULT_MODELS = list(OPT_HIDDEN_DIMS)
DEFAULT_CAPS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256]
DEFAULT_CHECKPOINTS = [0, 50, 100, 200, 300, 500, 750, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--caps", type=float, nargs="+", default=DEFAULT_CAPS)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-root", default="experiments/outputs/raw")
    parser.add_argument(
        "--aggregate-output",
        default="experiments/outputs/aggregate_learncap_recovery/opt_fn_recovery_summary.csv",
    )
    parser.add_argument(
        "--minimal-output",
        default="experiments/outputs/aggregate_learncap_recovery/opt_fn_recovery_minimal_by_d.csv",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-train-tokens", type=int, default=524288)
    parser.add_argument("--max-eval-tokens", type=int, default=131072)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--train-scope", choices=["all", "target_caps", "caps_only"], default="target_caps")
    parser.add_argument("--optimizer", choices=["adamw", "sgd", "adafactor"], default="adafactor")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--cap-norm-dtype", choices=["float32", "native"], default="float32")
    parser.add_argument("--baseline-selection", choices=["final", "best"], default="best")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-weight-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-on-recovery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-after-first-recovered", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Run commands. Without this flag, only print them.")
    parser.add_argument("--skip-aggregate", action="store_true")
    return parser.parse_args()


def quote(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def safe_model_name(model: str) -> str:
    return model.replace("/", "_").replace("-", "_").replace(".", "p")


def format_number(value: float | int) -> str:
    return f"{value:g}"


def recovery_command(args: argparse.Namespace, model: str) -> list[str]:
    hidden = OPT_HIDDEN_DIMS.get(model, "unknown")
    output = Path(args.output_root) / f"opt_fn_recovery_{safe_model_name(model)}_d{hidden}.json"
    command = [
        args.python,
        "-m", "quantum_friendly_transformer.experiments.runners.recovery.wikitext_lm",
        "--model",
        model,
        "--caps",
        *[format_number(cap) for cap in args.caps],
        "--checkpoints",
        *[str(step) for step in args.checkpoints],
        "--include-baseline",
        "--baseline-selection",
        args.baseline_selection,
        "--relative-tolerance",
        format_number(args.relative_tolerance),
        "--seq-len",
        str(args.seq_len),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--learning-rate",
        format_number(args.learning_rate),
        "--weight-decay",
        format_number(args.weight_decay),
        "--warmup-ratio",
        format_number(args.warmup_ratio),
        "--max-train-tokens",
        str(args.max_train_tokens),
        "--max-eval-tokens",
        str(args.max_eval_tokens),
        "--train-scope",
        args.train_scope,
        "--optimizer",
        args.optimizer,
        "--dtype",
        args.dtype,
        "--cap-norm-dtype",
        args.cap_norm_dtype,
        "--max-grad-norm",
        format_number(args.max_grad_norm),
        "--device",
        args.device,
        "--output",
        str(output),
    ]
    if args.cache_dir:
        command.extend(["--cache-dir", args.cache_dir])
    if args.early_stop_on_recovery:
        command.append("--early-stop-on-recovery")
    if args.stop_after_first_recovered:
        command.append("--stop-after-first-recovered")
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    if args.low_cpu_mem_usage:
        command.append("--low-cpu-mem-usage")
    if args.fast_weight_stats:
        command.append("--fast-weight-stats")
    if args.attn_implementation:
        command.extend(["--attn-implementation", args.attn_implementation])
    if args.offline:
        command.append("--offline")
    return command


def aggregate_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        "-m", "quantum_friendly_transformer.experiments.runners.reports.aggregate_learnable_cap_recovery",
        "--input-glob",
        str(Path(args.output_root) / "opt_fn_recovery_*.json"),
        "--output",
        args.aggregate_output,
        "--minimal-output",
        args.minimal_output,
    ]


def main() -> None:
    args = parse_args()
    commands = [recovery_command(args, model) for model in args.models]
    if not args.skip_aggregate:
        commands.append(aggregate_command(args))

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {quote(command)}", flush=True)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
