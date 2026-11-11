"""Resolve saved recovery states after relocating an experiment directory."""

from pathlib import Path


def recovery_checkpoint_path(source: Path, reported: str) -> Path:
    """Require the matching sibling checkpoint, ignoring its historical root."""
    expected = source.with_name(f"{source.stem}.fn24.recovery.pt")
    if not reported or Path(reported).name != expected.name:
        raise ValueError(f"recovery checkpoint for {source} is not its matching sibling: {reported}")
    if not expected.is_file():
        raise FileNotFoundError(f"missing recovery checkpoint for {source}: {expected}")
    return expected
