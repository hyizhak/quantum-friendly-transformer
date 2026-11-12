"""Package completed experiment records without model weights or raw datasets.

Each --cohort NAME DIRECTORY produces a deterministic ZIP of numeric records.
Optional --path-map JSON supplies literal path substitutions for portable
metadata. The inventory distinguishes source bytes from exported bytes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile


JSON_RECORDS = {
    "result.json", "config.json", "summary.json", "selection.json",
    "validation.json", "selected_scale_stats.json", "parameter_inventory.json",
    "first_step_gradient_audit.json", "independent_forward_checks.json",
    "uniform_gradient_check.json",
}
LOCAL_METADATA = {"cache_dir", "snapshot_path", "extracted_path"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def selected(path, root):
    relative = path.relative_to(root)
    if path.is_symlink() or "source_snapshot" in relative.parts:
        return False
    if any(part.startswith("private_") for part in relative.parts):
        return False
    return path.suffix == ".csv" or (
        path.suffix == ".json" and (
            path.name in JSON_RECORDS or "records" in relative.parts
            or path.name.endswith("_config.json") or path.name.startswith("config_seed")
        )
    )


def portable_string(value, replacements):
    for source, destination in replacements:
        value = value.replace(source, destination)
    # Unrelated local cache paths are bookkeeping, not scientific inputs.
    if value.startswith("/"):
        return "external/" + Path(value).name
    return value


def portable_json(value, replacements):
    if isinstance(value, dict):
        return {
            portable_string(k, replacements): portable_json(v, replacements)
            for k, v in value.items() if k not in LOCAL_METADATA
        }
    if isinstance(value, list):
        return [portable_json(v, replacements) for v in value]
    if isinstance(value, str):
        return portable_string(value, replacements)
    return value


def exported_bytes(path, replacements):
    data = path.read_bytes()
    if path.suffix == ".json":
        value = json.loads(data)
        if isinstance(value, dict) and (value.get("smoke") or value.get("workflow_validation")):
            raise ValueError(f"Validation-only data is not a completed experiment: {path}")
        changed = portable_json(value, replacements)
        if changed != value:
            return data, (json.dumps(changed, indent=2, ensure_ascii=False) + "\n").encode()
    elif any(source.encode() in data for source, _ in replacements):
        rows = list(csv.reader(io.StringIO(data.decode())))
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerows([[portable_string(cell, replacements) for cell in row] for row in rows])
        return data, output.getvalue().encode()
    return data, data


def pack(cohorts, output, replacements=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inventory = {
        "format_version": 1,
        "scope": "Completed experiment metrics, predictions, and figure source data.",
        "omitted": ["model weights", "raw tensor datasets", "source snapshots", "local caches"],
        "metadata": (
            "Scientific values are preserved. Paths are portable collection references; "
            "local cache fields are omitted. source_sha256 identifies the original bytes; "
            "sha256 identifies the distributed bytes. Recorded checkpoint/source digests "
            "refer to the original training artifacts, not the reorganized source code."
        ),
        "cohorts": [],
    }
    names = set()
    for name, root in cohorts:
        if not re.fullmatch(r"[a-z0-9_]+", name) or name in names:
            raise ValueError(f"Invalid or repeated cohort name: {name}")
        names.add(name)
        root = Path(root).resolve(strict=True)
        archive = output / (name + ".zip")
        members = []
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for path in sorted(root.rglob("*")):
                if not path.is_file() or not selected(path, root):
                    continue
                original, data = exported_bytes(path, replacements)
                member = path.relative_to(root).as_posix()
                info = zipfile.ZipInfo(member, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                zf.writestr(info, data, compresslevel=9)
                members.append({
                    "path": member, "bytes": len(data), "sha256": digest(data),
                    "source_sha256": digest(original), "metadata_normalized": original != data,
                })
        if not members:
            raise ValueError(f"No result records in {root}")
        inventory["cohorts"].append({
            "name": name, "archive": archive.name, "bytes": archive.stat().st_size,
            "sha256": digest(archive.read_bytes()), "members": members,
        })
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return inventory


def verify(directory):
    directory = Path(directory)
    inventory = json.loads((directory / "inventory.json").read_text())
    checked = 0
    for cohort in inventory["cohorts"]:
        archive = directory / cohort["archive"]
        if digest(archive.read_bytes()) != cohort["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {archive}")
        with zipfile.ZipFile(archive) as zf:
            expected = {m["path"] for m in cohort["members"]}
            if set(zf.namelist()) != expected or len(zf.namelist()) != len(expected):
                raise ValueError(f"Archive inventory mismatch: {archive}")
            for member in cohort["members"]:
                data = zf.read(member["path"])
                if len(data) != member["bytes"] or digest(data) != member["sha256"]:
                    raise ValueError(f"Record checksum mismatch: {archive}:{member['path']}")
                checked += 1
    return {"status": "passed", "cohorts": len(inventory["cohorts"]), "records": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", nargs=2, action="append", metavar=("NAME", "DIRECTORY"))
    parser.add_argument("--path-map", type=Path, help="JSON object of literal metadata path replacements")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path, help="Verify an existing data directory")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify), indent=2))
    else:
        if not args.cohort or not args.output:
            parser.error("packaging requires --cohort and --output")
        replacements = json.loads(args.path_map.read_text()) if args.path_map else {}
        pack(args.cohort, args.output, sorted(replacements.items(), key=lambda x: -len(x[0])))
        print(json.dumps(verify(args.output), indent=2))


if __name__ == "__main__":
    main()
