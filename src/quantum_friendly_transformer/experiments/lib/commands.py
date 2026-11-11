"""Config and command helpers shared by experiment launchers."""

from __future__ import annotations

import shlex
from pathlib import Path


def load_config(path: Path) -> dict:
    import yaml
    from quantum_friendly_transformer.experiments.cli import config_text

    data = yaml.safe_load(config_text(path))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def parse_scalar(value: str):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def apply_overrides(config: dict, overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be KEY=VALUE: {override!r}")
        key, value = override.split("=", 1)
        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = parse_scalar(value)


def arg_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def extend_arg(command: list[str], key: str, value) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(arg_name(key))
        return
    command.append(arg_name(key))
    if isinstance(value, (list, tuple)):
        command.extend(str(item) for item in value)
    else:
        command.append(str(value))


def merged_run_configs(config: dict) -> list[dict]:
    defaults = config.get("defaults", {})
    if defaults and not isinstance(defaults, dict):
        raise ValueError("config field 'defaults' must be a mapping")
    runs = config.get("runs")
    if runs is None:
        return [config]
    if not isinstance(runs, list):
        raise ValueError("config field 'runs' must be a list")

    merged = []
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("each run entry must be a mapping")
        item = {key: value for key, value in defaults.items() if key != "args"}
        item.update({key: value for key, value in run.items() if key != "args"})
        args = {}
        args.update(defaults.get("args", {}))
        args.update(run.get("args", {}))
        item["args"] = args
        merged.append(item)
    return merged


def quote(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)
