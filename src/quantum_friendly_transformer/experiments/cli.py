"""Run packaged research workflows without a repository checkout."""

from __future__ import annotations

import argparse
from importlib import import_module
from importlib.resources import files
from pathlib import Path
import sys


def available_runners() -> list[str]:
    root = files("quantum_friendly_transformer.experiments").joinpath("runners")
    runners = ["suite"]
    for group in ("diagnostics", "normalisation", "recovery", "reports"):
        runners.extend(
            f"{group}.{item.name[:-3]}"
            for item in root.joinpath(group).iterdir()
            if item.name.endswith(".py") and item.name != "__init__.py"
        )
    return sorted(runners)


def available_configs() -> list[str]:
    root = files("quantum_friendly_transformer.experiments").joinpath("configs")
    names = [item.name for item in root.iterdir() if item.name.endswith(".yaml")]
    names.extend(
        f"recovery/{item.name}" for item in root.joinpath("recovery").iterdir()
        if item.name.endswith(".yaml")
    )
    return sorted(names)


def config_text(name: str | Path) -> str:
    """Read a local YAML file or a named configuration shipped in the wheel."""
    path = Path(name)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    key = str(name).removeprefix("experiments/configs/")
    if key not in available_configs():
        raise FileNotFoundError(f"Unknown config {name!r}; run qft-experiment --list-configs")
    return files("quantum_friendly_transformer.experiments").joinpath(
        "configs", *key.split("/")
    ).read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List installed experiment runners.")
    parser.add_argument("--list-configs", action="store_true", help="List packaged YAML configs.")
    parser.add_argument("runner", nargs="?", help="For example diagnostics.forward_stats")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(available_runners()))
        return
    if args.list_configs:
        print("\n".join(available_configs()))
        return
    if args.runner not in available_runners():
        parser.error("Choose an installed runner; use --list to see names.")
    module = f"quantum_friendly_transformer.experiments.runners.{args.runner}"
    previous = sys.argv
    try:
        sys.argv = [module, *args.args]
        import_module(module).main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
