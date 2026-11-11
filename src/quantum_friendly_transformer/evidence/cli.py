"""List, extract, or verify packaged experiment evidence."""

from __future__ import annotations

import argparse
import json
import zipfile

from .store import EvidenceStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "extract", "verify"])
    parser.add_argument("--group", help="Filter by an evidence group, e.g. sst2 or depth.")
    parser.add_argument("--root", help="Existing directory containing experiments/outputs/.")
    parser.add_argument("--bundle", help="Companion evidence ZIP with measurements and checkpoints.")
    parser.add_argument("--destination", default=".", help="Extraction directory (default: current directory).")
    parser.add_argument("--include-bulk", action="store_true", help="Also extract or verify companion artifacts.")
    parser.add_argument("--json", action="store_true", help="Print the catalog entries as JSON when listing.")
    args = parser.parse_args(argv)
    try:
        store = EvidenceStore(root=args.root, bundle=args.bundle)
        artifacts = store.list_artifacts(args.group)
        if args.command == "list":
            if args.json:
                from dataclasses import asdict
                print(json.dumps([asdict(item) for item in artifacts], indent=2))
            else:
                for item in artifacts:
                    location = "wheel" if item.bundled else "companion"
                    print(f"{location:9s} {item.size:12,d}  {item.path}")
                print(f"{len(artifacts)} artifacts; groups: {', '.join(store.groups)}")
        elif args.command == "extract":
            count = sum(1 for _ in store.extract(
                args.destination, group=args.group, include_bulk=args.include_bulk,
            ))
            print(f"Extracted and verified {count} artifacts into {args.destination}")
        else:
            failures = store.verify(group=args.group, include_bulk=args.include_bulk)
            count = len(store.list_artifacts(args.group, bundled=None if args.include_bulk else True))
            for name, reason in failures.items():
                print(f"FAILED {name}: {reason}")
            print(f"Verified {count - len(failures)}/{count} artifacts")
            return int(bool(failures))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
