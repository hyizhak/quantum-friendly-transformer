#!/usr/bin/env python
"""Run FN recovery experiments from one config-driven entrypoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from quantum_friendly_transformer.experiments.lib.commands import apply_overrides, extend_arg, load_config, merged_run_configs, quote


MODULE_BY_PROTOCOL = {
    "glue_encoder": "glue_encoder",
    "glue_sequence_classifier": "glue_sequence_classifier",
    "glue_decoder": "glue_sequence_classifier",
    "wikitext_lm": "wikitext_lm",
    "glue_warmstart": "glue_warmstart",
    "glue_distill": "glue_distill",
    "opt_wikitext_sweep": "opt_wikitext_sweep",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--python", default=None, help="Python executable. Defaults to the current interpreter.")
    parser.add_argument("--execute", action="store_true", help="Run the command. Without this flag, only print it.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a scalar config arg. Use dotted keys under args, e.g. args.output=out.json.",
    )
    return parser.parse_args()


def build_command(run_config: dict, python: str) -> list[str]:
    protocol = run_config.get("protocol")
    if protocol not in MODULE_BY_PROTOCOL:
        allowed = ", ".join(sorted(MODULE_BY_PROTOCOL))
        raise ValueError(f"unknown protocol {protocol!r}; expected one of: {allowed}")

    args = run_config.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("config field 'args' must be a mapping")

    module = f"quantum_friendly_transformer.experiments.runners.recovery.{MODULE_BY_PROTOCOL[protocol]}"
    command = [python, "-m", module]
    for key, value in args.items():
        extend_arg(command, key, value)
    return command


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args.overrides)
    python = args.python or config.get("python") or sys.executable
    commands = [build_command(run_config, python) for run_config in merged_run_configs(config)]
    for index, command in enumerate(commands, start=1):
        prefix = f"[{index}/{len(commands)}] " if len(commands) > 1 else ""
        print(prefix + quote(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
