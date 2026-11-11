#!/usr/bin/env python
"""Remove generated experiment outputs outside the retained report scope."""

from __future__ import annotations

import argparse
from pathlib import Path

from quantum_friendly_transformer.experiments.lib.retained import (
    RETAINED_FIGURE_PREFIXES,
    RETAINED_OUTPUT_DIRECTORIES,
    RETAINED_RAW_OUTPUTS,
)


RAW_KEEP = RETAINED_RAW_OUTPUTS
RETAINED_DIRECTORIES = RETAINED_OUTPUT_DIRECTORIES
FIGURE_PREFIXES = RETAINED_FIGURE_PREFIXES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/outputs"))
    parser.add_argument("--execute", action="store_true", help="Delete files. Without this flag, only print a dry run.")
    parser.add_argument("--list-delete", action="store_true", help="Print every file that would be deleted.")
    return parser.parse_args()


def retained_files(root: Path) -> set[str]:
    keep = set(RAW_KEEP)
    for rel in list(RAW_KEEP):
        path = root / rel
        metadata = Path(str(path) + ".metadata.json")
        if metadata.exists():
            keep.add(str(metadata.relative_to(root)))

    for dirname in RETAINED_DIRECTORIES:
        directory = root / dirname
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                keep.add(str(path.relative_to(root)))

    figure_dir = root / "figures"
    if figure_dir.exists():
        for path in figure_dir.iterdir():
            if path.is_file() and path.name.startswith(FIGURE_PREFIXES):
                keep.add(str(path.relative_to(root)))
    return keep


def planned_deletions(root: Path, keep: set[str]) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and str(path.relative_to(root)) not in keep)


def missing_retained(root: Path, keep: set[str]) -> list[str]:
    return sorted(rel for rel in keep if not (root / rel).exists())


def remove_empty_dirs(root: Path) -> int:
    removed = 0
    if not root.exists():
        return removed
    directories = sorted((path for path in root.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True)
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            continue
        removed += 1
    return removed


def main() -> None:
    args = parse_args()
    root = args.output_root
    keep = retained_files(root)
    delete = planned_deletions(root, keep)
    missing = missing_retained(root, keep)

    print(f"output_root={root}")
    print(f"retained_files={len(keep)}")
    print(f"delete_files={len(delete)}")
    print(f"missing_retained_files={len(missing)}")
    if missing:
        for rel in missing:
            print(f"MISSING {rel}")
    if args.list_delete:
        for path in delete:
            print(f"DELETE {path.relative_to(root)}")

    if not args.execute:
        print("dry_run=true")
        return

    for path in delete:
        path.unlink()
    removed_dirs = remove_empty_dirs(root)
    print(f"deleted_files={len(delete)}")
    print(f"removed_empty_dirs={removed_dirs}")


if __name__ == "__main__":
    main()
