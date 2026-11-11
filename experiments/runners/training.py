#!/usr/bin/env python
"""Run legacy trainer recipes from config files."""

from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

for _qft_parent in _Path(__file__).resolve().parents:
    if (_qft_parent / "src").exists() and (_qft_parent / "experiments").exists():
        _sys.path.insert(0, str(_qft_parent))
        _sys.path.insert(0, str(_qft_parent / "src"))
        break

import argparse
import subprocess
import sys
from pathlib import Path

from experiments.lib.commands import load_config, quote
from quantum_friendly_transformer.trainer.legacy_recipes import (
    LEGACY_TRAINER_RECIPES,
    legacy_trainer_command,
    trainer_recipe_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--python", default=None, help="Python executable. Defaults to the current interpreter.")
    parser.add_argument("--execute", action="store_true", help="Run commands. Without this flag, only print them.")
    parser.add_argument("--list-recipes", action="store_true")
    return parser.parse_args()


def command_from_run(run: dict, python: str) -> list[str]:
    recipe = run.get("recipe")
    if not recipe:
        raise ValueError("training config run must define 'recipe'")
    extra_args = run.get("extra_args") or []
    if not isinstance(extra_args, list):
        raise ValueError("'extra_args' must be a list")
    return legacy_trainer_command(recipe, python=python, extra_args=[str(item) for item in extra_args])


def run_configs(config: dict) -> list[dict]:
    runs = config.get("runs")
    if runs is None:
        return [config]
    if not isinstance(runs, list):
        raise ValueError("config field 'runs' must be a list")
    return runs


def print_recipes() -> None:
    for name in trainer_recipe_names():
        recipe = LEGACY_TRAINER_RECIPES[name]
        print(f"{name}: {recipe.description}")


def main() -> None:
    args = parse_args()
    if args.list_recipes:
        print_recipes()
        return
    if args.config is None:
        raise SystemExit("--config is required unless --list-recipes is used")
    config = load_config(args.config)
    python = args.python or config.get("python") or sys.executable
    commands = [command_from_run(run, python) for run in run_configs(config)]
    for index, command in enumerate(commands, start=1):
        prefix = f"[{index}/{len(commands)}] " if len(commands) > 1 else ""
        print(prefix + quote(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
