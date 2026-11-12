"""Verify reported SST-2 scores and source provenance in a relocated release.

Uses the standard library and compact records only. This checks recorded
predictions, not a fresh model inference; checkpoint replay is a separate audit.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def local_source(recorded, source_root=ROOT):
    parts = Path(recorded).parts
    # Prefer an explicit repository-relative anchor, never the old machine path.
    for anchor in ("src", "experiments", "results", "tools"):
        if anchor in parts:
            return source_root.joinpath(*parts[parts.index(anchor):])
    raise ValueError(f"Cannot relocate source record: {recorded}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    args = parser.parse_args()
    results = args.results.resolve()
    completed = []
    source_hashes = {}
    for manifest_path in sorted(results.rglob("manifest.json")):
        result_path = manifest_path.with_name("result.json")
        if not result_path.exists():
            continue
        record = json.loads(result_path.read_text())
        if record.get("smoke") or "historical_recovery_threshold" not in record:
            continue
        manifest = json.loads(manifest_path.read_text())
        for name, digest in manifest["source_sha256"].items():
            source = local_source(name, args.source_root.resolve())
            if source not in source_hashes:
                source_hashes[source] = hashlib.sha256(source.read_bytes()).hexdigest()
            if source_hashes[source] != digest:
                raise ValueError(f"Training source hash mismatch: {source}")
        predictions = rows(result_path.with_name("predictions.csv"))
        correct = sum(row["label"] == row["prediction"] for row in predictions)
        if len(predictions) != 872 or correct != record["evaluation"]["correct"]:
            raise ValueError(f"Prediction counts disagree: {result_path}")
        accuracy = correct / len(predictions)
        if not math.isclose(accuracy, record["evaluation"]["accuracy"], abs_tol=1e-7):
            raise ValueError(f"Accuracy disagrees: {result_path}")
        recovered = accuracy >= record["historical_recovery_threshold"]
        if recovered != record["meets_historical_threshold"]:
            raise ValueError(f"Recovery decision disagrees: {result_path}")
        completed.append({"cohort": result_path.parent.parent.name, "name": record["name"],
                          "model": record["model"], "mode": record["mode"], "accuracy": accuracy,
                          "recovered": recovered})
    for group in rows(results / "recovery_extension_comparison.csv"):
        cohort = {"initial": "recovery_extension", "lower_lr": "recovery_extension_low_lr",
                  "replication_3m": "recovery_positive_control"}[group["cohort"]]
        data = [r for r in completed if (r["cohort"], r["name"], r["mode"]) == (cohort, group["name"], group["mode"])]
        values = [r["accuracy"] for r in data]
        if len(values) != int(group["training_seeds"]) or not math.isclose(statistics.mean(values), float(group["mean_accuracy"]), abs_tol=1e-12):
            raise ValueError(f"Fixed-cohort mean/count mismatch: {group}")
        if len(values) > 1 and not math.isclose(statistics.stdev(values), float(group["std_accuracy"]), abs_tol=1e-12):
            raise ValueError(f"Fixed-cohort standard deviation mismatch: {group}")
        if sum(r["recovered"] for r in data) != int(group["seeds_meeting_historical_threshold"]):
            raise ValueError(f"Fixed-cohort success count mismatch: {group}")
    historical = {r["model"]: float(r["best_validation_accuracy"]) >= float(r["recovery_threshold"])
                  for r in rows(results / "new_code_audit/recovery_comparison.csv") if r["regime"] == "bounded"}
    recovered = {model for model, success in historical.items() if success}
    recovered.update(r["model"] for r in completed if r["mode"] == "bounded" and r["recovered"])
    story = rows(results / "recovery_story.csv")
    if set(historical) != {r["model"] for r in story}:
        raise ValueError("Original model roster disagrees")
    if recovered != {r["model"] for r in story if r["recovered_in_either"] == "True"}:
        raise ValueError("Distinct recovery count disagrees")
    print(json.dumps({
        "completed_bounded_records_verified": sum(r["mode"] == "bounded" for r in completed),
        "completed_uniform_records_verified": sum(r["mode"] == "uniform" for r in completed),
        "distinct_training_source_files_verified": len(source_hashes),
        "fixed_cohort_rows_verified": len(rows(results / "recovery_extension_comparison.csv")),
        "recovered_original_models": len(recovered), "original_models": len(historical),
        "scope": "Recorded prediction and provenance verification; no training or fresh model inference",
    }, indent=2))


if __name__ == "__main__":
    main()
