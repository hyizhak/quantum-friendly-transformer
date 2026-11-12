"""Extract matrix-level evidence for Comment 4(e), without loading or training models."""

import csv
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/reviewer"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    destination = RESULTS / "release"
    destination.mkdir(exist_ok=True)
    initial = []
    with zipfile.ZipFile(ROOT / "src/quantum_friendly_transformer/evidence/records.zip") as archive:
        for item in read_csv(RESULTS / "new_code_audit/recovery_comparison.csv"):
            if item["regime"] != "bounded":
                continue
            record = json.loads(archive.read(item["source"]))
            result = record["results"][0]
            after = {r["name"]: r for r in result["weight_stats_after_cap"]}
            for row in result["weight_stats_before"]:
                if row["name"] not in after:
                    raise ValueError(f"Missing post-cap norm: {row['name']}")
                raw, effective = row["frobenius_norm"], after[row["name"]]["frobenius_norm"]
                initial.append({
                    "model": item["model"], "phase": "historical_initial_before_training",
                    "matrix": row["name"], "shape": row["shape"],
                    "raw_frobenius_norm": raw, "effective_frobenius_norm": effective,
                    "raw_above_24": raw > 24,
                    "reduction_greater_than_one_percent": effective < .99 * raw,
                    "source": item["source"],
                })
    selected = []
    summary = []
    for item in read_csv(RESULTS / "recovery_story.csv"):
        if not item["new_best_result_path"]:
            continue
        run = Path(item["new_best_result_path"]).parent.name
        cohort = RESULTS / item["new_best_cohort"]
        matches = []
        for path in sorted(cohort.glob("selected_weight_norms*.csv")):
            data = [r for r in read_csv(path) if r["run"] == run]
            if data:
                matches.append((path, data))
        if not matches:
            raise ValueError(f"No selected matrix norms for {run}")
        path, data = matches[0]
        for row in data:
            selected.append({"model": item["model"], "cohort": item["new_best_cohort"],
                             **row, "source": path.relative_to(ROOT).as_posix()})
        summary.append({
            "model": item["model"], "run": run, "matrices": len(data),
            "active_caps": sum(r["cap_active"].lower() == "true" for r in data),
            "minimum_raw_frobenius": min(float(r["raw_frobenius_norm"]) for r in data),
            "maximum_raw_frobenius": max(float(r["raw_frobenius_norm"]) for r in data),
            "maximum_effective_frobenius": max(float(r["effective_frobenius_norm"]) for r in data),
            "source": path.relative_to(ROOT).as_posix(),
        })
    save_csv(destination / "historical_initial_projection_norms.csv", initial)
    # Older audit versions may not contain every low-rank metadata field.
    columns = list(dict.fromkeys(k for row in selected for k in row))
    save_csv(destination / "selected_projection_norms.csv", [{k: row.get(k, "") for k in columns} for row in selected])
    save_csv(destination / "selected_cap_activity.csv", summary)
    mistral = [r for r in initial if "Mistral" in r["model"]]
    for group, rows in (("Mistral initial attention", [r for r in mistral if ".self_attn." in r["matrix"]]),
                        ("Mistral initial MLP", [r for r in mistral if ".mlp." in r["matrix"]])):
        print(group, "matrices", len(rows), "raw above 24", sum(r["raw_above_24"] for r in rows),
              "raw range", min(r["raw_frobenius_norm"] for r in rows), max(r["raw_frobenius_norm"] for r in rows))
    print(f"Extracted {len(initial)} historical initial and {len(selected)} selected projection records.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
