"""Build the bundled records, evidence catalog, and optional companion ZIP.

Run from a checkout with the original experiment outputs. This preserves file
contents, including recorded failures and historical provenance. No models run.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import zipfile


OUTPUT_SUFFIXES = {".csv", ".json", ".parquet", ".npy", ".pt", ".png", ".pdf", ".md"}


def evidence_paths(root: Path) -> list[Path]:
    paths = [
        path for path in (root / "experiments/outputs").rglob("*")
        if path.is_file() and path.suffix in OUTPUT_SUFFIXES
    ]
    paths += list((root / "results").glob("*.md"))
    paths += list((root / "experiments/reports").glob("*.md"))
    response = root / "response.tex"
    if response.exists():
        paths.append(response)
    return sorted(paths)


def belongs_in_wheel(relative: Path) -> bool:
    if relative.suffix in {".json", ".md", ".tex"}:
        return True
    return relative.suffix == ".csv" and "raw" not in relative.parts


def artifact_groups(relative: Path) -> list[str]:
    text = relative.as_posix().lower()
    groups = set()
    if "sst2" in text or "attention_recovery" in text or "recovery_fn_vs_d" in text:
        groups.add("sst2")
    if "n_scaling" in text or "z_over" in text or "aggregate_30" in text or relative.name.startswith("diag_"):
        groups.add("partition")
    if "layernorm" in text or "denom" in text or "norm_quantiles" in text or relative.name.startswith("diag_"):
        groups.add("layernorm")
    if any(part in text for part in ("across_layers", "token_norm", "attention_input", "depth_stability", "norm_by_layer", "shared_attention_norm")):
        groups.add("depth")
    if relative.suffix == ".pt":
        groups.add("checkpoints")
    if relative.suffix in {".png", ".pdf"}:
        groups.add("figures")
    if relative.name.endswith("metadata.json"):
        groups.add("metadata")
    if relative.suffix in {".md", ".tex"}:
        groups.add("reports")
    if relative.suffix == ".csv" and "raw" not in relative.parts:
        groups.add("tables")
    if relative.suffix == ".json" and not relative.name.endswith("metadata.json"):
        groups.add("runs")
    if relative.suffix in {".parquet", ".npy"} or (relative.suffix == ".csv" and "raw" in relative.parts):
        groups.add("measurements")
    return sorted(groups or {"other"})


def write_entry(archive, source: Path, name: str, bundled: bool) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with ExitStack() as stack:
        stream = stack.enter_context(source.open("rb"))
        output = None
        if archive is not None:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED if bundled else zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            output = stack.enter_context(archive.open(info, "w", force_zip64=True))
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
            if output is not None:
                output.write(block)
    return size, digest.hexdigest()


def build(root: Path, package: Path, bundle: Path | None) -> dict:
    paths = evidence_paths(root)
    if not paths or not any(path.suffix == ".pt" for path in paths):
        raise FileNotFoundError("Build requires the original experiment outputs, including checkpoints")
    package.mkdir(parents=True, exist_ok=True)
    records_path = package / "records.zip"
    if bundle is not None:
        if bundle.exists():
            raise FileExistsError(f"Refusing to overwrite an evidence bundle: {bundle}")
        bundle.parent.mkdir(parents=True, exist_ok=True)
    names = {path.relative_to(root).as_posix() for path in paths}
    entries = []
    with ExitStack() as stack:
        records = stack.enter_context(zipfile.ZipFile(records_path, "w", compression=zipfile.ZIP_DEFLATED))
        bulk = stack.enter_context(zipfile.ZipFile(bundle, "x")) if bundle is not None else None
        for index, path in enumerate(paths, 1):
            relative = path.relative_to(root)
            name = relative.as_posix()
            bundled = belongs_in_wheel(relative)
            size, digest = write_entry(records if bundled else bulk, path, name, bundled)
            related = []
            metadata = name + ".metadata.json"
            if metadata in names:
                related.append(metadata)
            if name.endswith(".fn24.recovery.pt"):
                source = name.removesuffix(".fn24.recovery.pt") + ".json"
                if source in names:
                    related.append(source)
            checkpoint = name.removesuffix(".json") + ".fn24.recovery.pt"
            if checkpoint in names:
                related.append(checkpoint)
            entries.append({
                "path": name, "size": size, "sha256": digest,
                "bundled": bundled, "groups": artifact_groups(relative), "related": related,
            })
            if index % 100 == 0:
                print(f"Catalogued {index}/{len(paths)} files", flush=True)
    catalog = {
        "schema_version": 1,
        "description": "Recorded research outputs; inclusion does not imply successful recovery or complete reviewer evidence.",
        "layout": "Original paths relative to the repository root; compact records ship in records.zip, other entries in the companion ZIP.",
        "groups": "Filename-based discovery labels; consult each record and metadata for measured fields and sampling.",
        "checkpoints": "Recovery checkpoints contain trainable_state_dict and require the corresponding base model and adapters.",
        "response": "response.tex is the historical response draft, not a certification that its claims are supported.",
        "artifact_count": len(entries),
        "bundled_count": sum(entry["bundled"] for entry in entries),
        "artifacts": entries,
    }
    (package / "catalog.json").write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    if bundle is not None:
        digest = hashlib.sha256()
        with bundle.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        bundle.with_suffix(bundle.suffix + ".sha256").write_text(f"{digest.hexdigest()}  {bundle.name}\n")
    print(f"Bundled {catalog['bundled_count']}/{len(entries)} artifacts ({records_path.stat().st_size:,} compressed bytes)")
    if bundle is not None:
        print(f"Companion: {bundle} ({bundle.stat().st_size:,} bytes)")
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--package-dir", type=Path, default=Path("src/quantum_friendly_transformer/evidence"))
    parser.add_argument("--bundle", type=Path, help="Also create the large companion ZIP at this path")
    args = parser.parse_args()
    build(args.source_root.resolve(), args.package_dir.resolve(), args.bundle)


if __name__ == "__main__":
    main()
