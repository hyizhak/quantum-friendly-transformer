"""Build and verify a source/results release without changing experiment files.

This is separate from the historical wheel evidence snapshot. Nothing is
uploaded. Omitted weights and missing historical companion files are explicit.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "RESEARCH_BUNDLE_MANIFEST.json"
SMALL_SUFFIXES = {
    ".py", ".json", ".csv", ".txt", ".log", ".tex", ".md", ".rst",
    ".toml", ".yaml", ".yml", ".mplstyle", ".sh", ".pdf", ".png",
}
def is_correspondence(path, root):
    relative = path.relative_to(root)
    if any(part.startswith("private_") or part == "correspondence" for part in relative.parts):
        return True
    tokens = set(re.split(r"[_.\/-]+", "/".join(relative.parts).lower()))
    if {"response", "responses", "reply", "addendum", "correspondence"} & tokens:
        return True
    return path.suffix == ".tex" and "table" not in path.stem.lower()


def archive_name(path, root, results=None):
    if results is not None and path.is_relative_to(results):
        return "results/" + path.relative_to(results).as_posix()
    return path.relative_to(root).as_posix()


def digest_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def digest_file(path):
    with path.open("rb") as stream:
        return digest_stream(stream)


def inventory(root, results=None, include_checkpoints=False):
    compact = set()
    for directory in ("src", "experiments", "tests", "tools", "docs", "examples"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and path.suffix in SMALL_SUFFIXES and not is_correspondence(path, root):
                if not any(p in {"__pycache__", ".cache", "outputs", ".pytest_cache"} for p in path.parts):
                    compact.add(path)
    for name in ("README.md", "LICENSE", "pyproject.toml", "MANIFEST.in", "uv.lock", "environment.yml"):
        path = root / name
        if path.is_file():
            compact.add(path)
    compact.add(root / "src/quantum_friendly_transformer/evidence/records.zip")
    weights, omitted = set(), []
    if results is not None:
        if not results.is_dir():
            raise FileNotFoundError(results)
        compact.update(p for p in results.rglob("*") if p.is_file()
                       and p.suffix in SMALL_SUFFIXES and not is_correspondence(p, results))
        for path in sorted(results.rglob("*")):
            if not path.is_file() or path.suffix not in {".pt", ".pth"} or is_correspondence(path, results):
                continue
            if include_checkpoints:
                weights.add(path)
            else:
                omitted.append({"path": archive_name(path, root, results), "bytes": path.stat().st_size,
                                "reason": "Enable --include-checkpoints to include saved tensors"})
    if not all(p.is_file() for p in compact | weights):
        raise FileNotFoundError("Required release file is missing")
    for path in compact | weights:
        if path.is_symlink() or not any(path.resolve().is_relative_to(base.resolve())
                                       for base in (root, results) if base is not None):
            raise ValueError(f"Release source must be a regular file inside the selected inputs: {path}")
    names = [archive_name(p, root, results) for p in compact | weights]
    if len(names) != len(set(names)):
        raise ValueError("Selected inputs contain conflicting archive names")
    return sorted(compact), sorted(weights), omitted


def write_source(archive, path, root, part, results=None):
    name = archive_name(path, root, results)
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.external_attr = 0o644 << 16
    info.compress_type = zipfile.ZIP_STORED if part == "checkpoints" else zipfile.ZIP_DEFLATED
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            target.write(block)
            digest.update(block)
            size += len(block)
    if part == "checkpoints" and path.name == "selected.pth":
        result = path.with_name("result.json")
        if result.exists():
            expected = json.loads(result.read_text())["checkpoint_sha256"]
            if digest.hexdigest() != expected:
                raise ValueError(f"Checkpoint differs from its recorded result: {name}")
    return {"path": name, "part": part, "bytes": size, "sha256": digest.hexdigest()}


def verify(paths):
    manifests = []
    total = 0
    parts_seen = set()
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            raw_manifest = archive.read(MANIFEST_NAME)
            manifest = json.loads(raw_manifest)
            manifests.append(raw_manifest)
            part = "checkpoints" if path.name.endswith("-checkpoints.zip") else "code-results"
            if part in parts_seen:
                raise ValueError("Duplicate release part")
            parts_seen.add(part)
            expected = {row["path"]: row for row in manifest["artifacts"] if row["part"] == part}
            if len(archive.namelist()) != len(set(archive.namelist())):
                raise ValueError("Duplicate ZIP entries")
            if set(archive.namelist()) != set(expected) | {MANIFEST_NAME}:
                raise ValueError(f"File inventory mismatch: {path}")
            for name, row in expected.items():
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError(f"Unsafe archive path: {name}")
                with archive.open(name) as stream:
                    digest = digest_stream(stream)
                if digest != row["sha256"] or archive.getinfo(name).file_size != row["bytes"]:
                    raise ValueError(f"Integrity mismatch: {name}")
            total += len(expected)
            print(f"Verified {len(expected)} files in {path.name}", flush=True)
    if len(set(manifests)) != 1 or parts_seen != {"code-results", "checkpoints"}:
        raise ValueError("Both matching release parts are required")
    return {"verified_artifacts": total, "parts": sorted(parts_seen), "all_sha256_match": True}


def source_commit(root):
    if not (root / ".git").exists():
        return None
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build(root, destination, name, results=None, include_checkpoints=False):
    compact, weights, omitted = inventory(root, results, include_checkpoints)
    estimate = sum(p.stat().st_size for p in compact + weights)
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < estimate + 2 * 1024**3:
        raise OSError("Insufficient room for archive plus a 2 GiB free-space reserve")
    paths = [destination / f"{name}-{part}.zip" for part in ("code-results", "checkpoints")]
    sidecars = [destination / f"{name}-manifest.json", destination / f"{name}-verification.json"]
    for path in paths + sidecars + [p.with_suffix(".zip.sha256") for p in paths]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    catalog = json.loads((root / "src/quantum_friendly_transformer/evidence/catalog.json").read_text())
    missing = [r for r in catalog["artifacts"] if not r["bundled"] and not (root / r["path"]).is_file()]
    entries = []
    with ExitStack() as stack:
        archives = [stack.enter_context(zipfile.ZipFile(p, "x", allowZip64=True)) for p in paths]
        for files, archive, part in zip((compact, weights), archives, ("code-results", "checkpoints")):
            for index, source in enumerate(files, 1):
                entries.append(write_source(archive, source, root, part, results))
                if index % (10 if part == "checkpoints" else 500) == 0:
                    print(f"Packed {index}/{len(files)} {part} files", flush=True)
        manifest = {
            "schema_version": 1,
            "name": name,
            "git_commit": source_commit(root),
            "source_policy": "Actual working-tree bytes, including uncommitted experiment code; per-file hashes identify the release",
            "scope": "Source code and explicitly selected experiment outputs; private correspondence excluded",
            "historical_companion": {"missing_artifacts": len(missing), "missing_bytes": sum(r["size"] for r in missing),
                                     "inventory": "src/quantum_friendly_transformer/evidence/catalog.json",
                                     "status": "Not included; historical compact records are preserved in records.zip"},
            "external_assets": "Pinned base models, tokenizers and SST-2/WikiText source data must be downloaded using the recorded revisions",
            "omitted_local_weights": omitted,
            "artifacts": entries,
        }
        payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        for archive in archives:
            info = zipfile.ZipInfo(MANIFEST_NAME, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    sidecars[0].write_bytes(payload)
    for path in paths:
        path.with_suffix(".zip.sha256").write_text(f"{digest_file(path)}  {path.name}\n")
    report = verify(paths)
    report.update(included_checkpoint_files=len(weights), omitted_local_weights=len(omitted),
                  historical_companion_missing=len(missing), archives=[{"name": p.name, "bytes": p.stat().st_size} for p in paths])
    sidecars[1].write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--name", default="qft-research-bundle")
    parser.add_argument("--verify", type=Path, nargs=2, metavar=("CODE_RESULTS", "CHECKPOINTS"))
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify), indent=2))
    else:
        if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in args.name):
            parser.error("--name must be a filename stem")
        if args.output is None:
            parser.error("--output is required when building a bundle")
        build(args.source.resolve(), args.output.resolve(), args.name,
              args.results.resolve() if args.results else None, args.include_checkpoints)


if __name__ == "__main__":
    main()
