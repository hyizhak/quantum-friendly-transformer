"""Recompute MQAR tables from saved predictions, without checkpoints or PyTorch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile


class Records:
    def __init__(self, path):
        self.path = Path(path)
        self.archive = zipfile.ZipFile(path) if self.path.is_file() else None
        self.hashes = {}

    def read(self, name):
        data = self.archive.read(name) if self.archive else (self.path / name).read_bytes()
        self.hashes[name] = hashlib.sha256(data).hexdigest()
        return data

    def json(self, name):
        return json.loads(self.read(name))

    def csv(self, name):
        return csv.DictReader(io.StringIO(self.read(name).decode()))

    def close(self):
        if self.archive:
            self.archive.close()


def aggregate(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def score(rows, width):
    counts = {"queries": 0, "correct": 0, "substituted_correct": 0,
              "prefix_queries": 0, "prefix_correct": 0, "prefix_substituted_correct": 0}
    keys = set()
    sequence_ids = set()
    alignment = hashlib.sha256()
    for row in rows:
        sequence, position, label = int(row["sequence"]), int(row["position"]), int(row["label"])
        key = (sequence, position)
        if key in keys:
            raise ValueError(f"Duplicate query: {key}")
        keys.add(key)
        sequence_ids.add(sequence)
        alignment.update(f"{sequence},{position},{label}\n".encode())
        correct = int(row["prediction"]) == label
        substituted = int(row["uniform_prediction"]) == label
        counts["queries"] += 1
        counts["correct"] += correct
        counts["substituted_correct"] += substituted
        if position + 1 >= width:
            counts["prefix_queries"] += 1
            counts["prefix_correct"] += correct
            counts["prefix_substituted_correct"] += substituted
    if not counts["queries"]:
        raise ValueError("Prediction file is empty")
    return {**counts, "sequences": len(sequence_ids), "query_alignment_sha256": alignment.hexdigest()}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(capped_path, standard_path, output):
    capped, standard = Records(capped_path), Records(standard_path)
    try:
        # Configured seeds and conditions determine the complete cohort.
        # Discover a run rather than assuming any particular seed exists.
        names = capped.archive.namelist() if capped.archive else [p.relative_to(capped.path).as_posix() for p in capped.path.glob("*/config.json")]
        first = sorted(n for n in names if n.startswith("native_seed") and n.endswith("/config.json"))[0]
        config = capped.json(first)
        seeds, width = config["seeds"], config["dimension"]
        conditions = config["test_conditions"]
        per_seed, norm_rows = [], []
        total_queries, steps = 0, set()
        expected_alignment = {}
        for label, store, variant in [
            ("capped", capped, "native"), ("average_trained", capped, "uniform"),
            ("standard", standard, "native"),
        ]:
            for seed in seeds:
                run = f"{variant}_seed{seed}"
                result = store.json(f"{run}/result.json")
                run_config = store.json(f"{run}/config.json")
                if result.get("smoke") or result.get("workflow_validation") or run_config.get("workflow_validation"):
                    raise ValueError("Expected completed experimental records, found a workflow validation run")
                for field in ("seeds", "dimension", "train_examples", "validation_examples", "test_examples", "train_data_seed", "validation_data_seed", "test_data_seed", "epochs", "batch_size", "test_conditions"):
                    if run_config[field] != config[field]:
                        raise ValueError(f"Unmatched experimental configuration: {field}")
                steps.add(result["training_steps"])
                saved = {(r["N"], r["pairs"]): r for r in result["evaluation"]}
                for condition in conditions:
                    length, pairs = condition["N"], condition["pairs"]
                    counts = score(store.csv(f"{run}/test_N{length}_K{pairs}_predictions.csv"), width)
                    expected = saved[length, pairs]
                    total_queries += counts["queries"]
                    for field in ("queries", "correct", "sequences"):
                        if counts[field] != expected[field]:
                            raise ValueError(f"Stored result disagrees with predictions: {run} {length} {field}")
                    if not math.isclose(counts["substituted_correct"] / counts["queries"], expected["uniform_ablation_accuracy"], abs_tol=1e-12):
                        raise ValueError(f"Stored substitution result disagrees with predictions: {run}")
                    alignment = counts["query_alignment_sha256"]
                    key = (length, pairs)
                    if expected_alignment.setdefault(key, alignment) != alignment:
                        raise ValueError(f"Queries are not matched across seeds/models: {key}")
                    for model, correct, prefix_correct in [(label, "correct", "prefix_correct")] + (
                        [("average_substituted", "substituted_correct", "prefix_substituted_correct")] if label == "capped" else []
                    ):
                        per_seed.append({
                            "model": model, "N": length, "pairs": pairs, "seed": seed,
                            "queries": counts["queries"], "correct": counts[correct],
                            "accuracy": counts[correct] / counts["queries"],
                            "prefix_queries": counts["prefix_queries"],
                            "prefix_correct": counts[prefix_correct],
                            "prefix_accuracy": counts[prefix_correct] / counts["prefix_queries"] if counts["prefix_queries"] else None,
                        })
                if label == "capped":
                    for phase in ("initial", "final"):
                        for row in store.csv(f"{run}/weights_{phase}.csv"):
                            if row["capped"] == "True":
                                norm_rows.append({"seed": seed, "phase": phase, **row})
        if len(steps) != 1:
            raise ValueError("Unequal training update budgets")
        rows = []
        for condition in conditions:
            for model in ("capped", "average_trained", "average_substituted", "standard"):
                group = [r for r in per_seed if r["model"] == model and r["N"] == condition["N"] and r["pairs"] == condition["pairs"]]
                stats = aggregate([r["accuracy"] for r in group])
                prefix = aggregate([r["prefix_accuracy"] for r in group]) if group[0]["prefix_queries"] else {"mean": None, "sd": None}
                rows.append({"model": model, "N": condition["N"], "pairs": condition["pairs"], "seeds": len(group),
                             "queries_per_seed": group[0]["queries"], "accuracy_mean": stats["mean"], "accuracy_sd": stats["sd"],
                             "prefix_queries_per_seed": group[0]["prefix_queries"], "prefix_accuracy_mean": prefix["mean"], "prefix_accuracy_sd": prefix["sd"]})
        primary = [r for r in rows if r["N"] == config["train_length"] and r["pairs"] == config["train_pairs"]]
        norms = []
        for projection in ("q", "k", "v", "o", "m1", "m2"):
            row = {"projection": projection}
            for phase in ("initial", "final"):
                group = [r for r in norm_rows if r["phase"] == phase and r["matrix"].split(".")[-1] == projection]
                row[phase + "_matrices"] = len(group)
                row[phase + "_active"] = sum(r["cap_active"] == "True" for r in group)
                for metric in ("raw_frobenius", "effective_frobenius"):
                    values = [float(r[metric]) for r in group]
                    row[f"{phase}_{metric}_min"], row[f"{phase}_{metric}_max"] = min(values), max(values)
            norms.append(row)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        for name, data in [("per_seed", per_seed), ("aggregate", rows), ("primary", primary), ("norms", norms), ("matrix_norms", norm_rows)]:
            write_csv(output / (name + ".csv"), data)
        summary = {"status": "passed", "cap": config["cap"], "dimension": width, "seeds": seeds,
                   "training_updates": steps.pop(), "prediction_queries_checked": total_queries,
                   "prefix_definition": "position + 1 >= dimension (positions are zero-based)",
                   "standard_deviation": "sample standard deviation across training seeds (ddof=1)",
                   "primary": primary, "results": rows, "norms": norms,
                   "source_sha256": {"capped": capped.hashes, "standard": standard.hashes},
                   "scope": "Reaggregation of completed predictions and recorded norms; no new training or checkpoint inference."}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        capped.close()
        standard.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capped-results", type=Path, required=True, help="Cohort ZIP or extracted directory")
    parser.add_argument("--standard-results", type=Path, required=True, help="Cohort ZIP or extracted directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.capped_results, args.standard_results, args.output)
    print(json.dumps({k: result[k] for k in ("status", "training_updates", "prediction_queries_checked", "primary")}, indent=2))


if __name__ == "__main__":
    main()
