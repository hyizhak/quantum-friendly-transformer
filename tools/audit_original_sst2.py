"""Trace the original retained SST-2 table to its own records and released code.

This does not train or reload a model. Later attention-controller and reviewer
recovery cohorts are excluded from the row mapping.
"""

import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/reviewer/original_sst2_audit"
PACKAGE = ROOT / "src/quantum_friendly_transformer"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest_path = PACKAGE / "experiments/lib/retained.py"
    tree = ast.parse(manifest_path.read_text())
    specs = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "RETAINED_RECOVERY_RUNS" for t in node.targets)
    )
    assert len(specs) == 9
    assert all("alpha_attention" not in row["source"] for row in specs)
    catalog_path = PACKAGE / "evidence/catalog.json"
    catalog = {r["path"]: r for r in json.loads(catalog_path.read_text())["artifacts"]}
    archive_path = PACKAGE / "evidence/records.zip"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    extracted = OUTPUT / "records"
    extracted.mkdir(exist_ok=True)
    source_records = {}
    rows, matrices = [], []

    with zipfile.ZipFile(archive_path) as archive:
        def read_record(name):
            data = archive.read(name)
            assert digest(data) == catalog[name]["sha256"], name
            destination = extracted / Path(name).name
            if destination.exists():
                assert destination.read_bytes() == data, destination
            else:
                destination.write_bytes(data)
            source_records[name] = {"sha256": digest(data), "extracted_path": str(destination.relative_to(ROOT))}
            return data

        aggregate_name = "experiments/outputs/aggregate_recovery_fn_vs_d/recovery_fn_vs_d_sst2.csv"
        aggregate = {
            row["model"]: row
            for row in csv.DictReader(io.StringIO(read_record(aggregate_name).decode()))
        }
        for spec in specs:
            name = "experiments/outputs/raw/" + spec["source"]
            data = json.loads(read_record(name))
            assert data["model"] == spec["model"]
            assert (data["seed"], data["split"], data["task"], data["max_length"], data["pad_to_max_length"]) == (42, "validation", "sst2", 128, True)
            assert data["max_train_samples"] is None and data["max_eval_samples"] is None
            run = next(r for r in data["results"] if r["cap"] == 24)
            initial = next(r for r in run["checkpoints"] if r["checkpoint_steps"] == 0)
            best = max(run["checkpoints"], key=lambda r: r["metric_value"])
            final = max(run["checkpoints"], key=lambda r: r["checkpoint_steps"])
            before = {r["name"]: r for r in run["weight_stats_before"]}
            after = {r["name"]: r for r in run["weight_stats_after_cap"]}
            gamma = {r["name"]: r["gamma"] for r in initial["cap_stats"]}
            assert set(before) == set(after) == set(gamma)
            assert len(before) == run["applied_caps"]
            short_name = spec["model"].split("/")[-1]
            current = []
            for weight_name, measurement in before.items():
                raw = measurement["frobenius_norm"]
                effective = after[weight_name]["frobenius_norm"]
                row = {
                    "model": short_name, "matrix": weight_name,
                    "group": "attention" if ".attn." in weight_name or ".self_attn." in weight_name else "ffn",
                    "initial_raw_frobenius": raw, "initial_effective_frobenius": effective,
                    "initial_gamma": gamma[weight_name],
                    "raw_exceeds_initial_gamma": raw > gamma[weight_name],
                    "recorded_norm_changed": abs(raw - effective) > 1e-6,
                    "cap_norm_dtype": data["cap_norm_dtype"], "source_record": name,
                }
                assert effective < 24, row
                current.append(row)
            matrices.extend(current)
            stored = aggregate[spec["model"]]
            assert int(stored["fn24_best_steps"]) == best["checkpoint_steps"]
            assert int(stored["fn24_final_steps"]) == final["checkpoint_steps"]
            assert abs(float(stored["fn24_best_metric"]) - best["metric_value"]) < 1e-12
            assert abs(float(stored["fn24_final_metric"]) - final["metric_value"]) < 1e-12
            baseline_path = spec.get("baseline_source")
            baseline = json.loads(read_record("experiments/outputs/raw/" + baseline_path)) if baseline_path else data
            baseline_run = baseline.get("baseline_run")
            baseline_trace = (baseline_run or {}).get("checkpoints", [])
            if baseline_trace:
                assert baseline["seed"] == 42 and baseline["split"] == "validation"
            original_target = spec.get("baseline_target", data.get("baseline_target"))
            assert abs(float(stored["baseline_target"]) - original_target) < 1e-12
            matching_configs = [
                str(p.relative_to(ROOT))
                for p in (PACKAGE / "experiments/configs/recovery").glob("*.yaml")
                if spec["source"] in p.read_text()
            ]
            rows.append({
                "model": short_name, "model_id": spec["model"], "d": spec["hidden_size_d"],
                "source_record": name, "baseline_source": baseline_path,
                "baseline_trace_present": bool(baseline_trace),
                "baseline_target": original_target,
                "run_baseline_target": data["baseline_target"], "run_threshold": data["threshold"],
                "baseline_reference_overridden": abs(original_target-data["baseline_target"])>1e-12,
                "fn24_best_accuracy": best["metric_value"],
                "fn24_best_step": best["checkpoint_steps"], "fn24_final_accuracy": final["metric_value"],
                "fn24_final_step": final["checkpoint_steps"], "seed": data["seed"],
                "train_scope": data["train_scope"], "trainable_parameters": run["trainable_params"],
                "optimizer": data["optimizer"], "learning_rate": data["learning_rate"],
                "weight_decay": data["weight_decay"], "batch_size": data["batch_size"],
                "grad_accum_steps": data["grad_accum_steps"], "dtype": data["dtype"],
                "cap_norm_dtype": data["cap_norm_dtype"], "target_matrices": len(current),
                "initial_raw_min": min(r["initial_raw_frobenius"] for r in current),
                "initial_raw_max": max(r["initial_raw_frobenius"] for r in current),
                "initial_effective_max": max(r["initial_effective_frobenius"] for r in current),
                "initial_gamma_min": min(gamma.values()), "initial_gamma_max": max(gamma.values()),
                "raw_above_initial_gamma": sum(r["raw_exceeds_initial_gamma"] for r in current),
                "initial_norm_changes": sum(r["recorded_norm_changed"] for r in current),
                "best_effective_norm_max": max(r["frobenius_norm"] for r in best["weight_stats_final"]),
                "attention_scaling_field_present": "attention_scaling" in data,
                "recorded_script_commit": data["script_commit"], "package_versions": data["package_versions"],
                "matching_released_configs": matching_configs,
            })

    code_paths = [manifest_path, catalog_path,
                  PACKAGE / "experiments/runners/recovery/glue_sequence_classifier.py",
                  PACKAGE / "experiments/runners/reports/aggregate_recovery_fn_vs_d.py",
                  PACKAGE / "norm_transformer/frobenius_cap.py",
                  PACKAGE / "norm_transformer/attention_scaling.py"]
    config_paths = sorted({name for row in rows for name in row["matching_released_configs"]})
    code_paths.extend(ROOT / name for name in config_paths)
    runner_name = "src/quantum_friendly_transformer/experiments/runners/recovery/glue_sequence_classifier.py"
    commit_checks = {}
    for commit in {row["recorded_script_commit"] for row in rows}:
        check = subprocess.run(["git", "cat-file", "-e", f"{commit}:{runner_name}"], cwd=ROOT, capture_output=True)
        commit_checks[commit] = {"contains_released_runner_path": check.returncode == 0}

    result = {
        "status": "passed", "original_table_rows": len(rows), "records_verified": len(source_records),
        "rows": rows, "source_records": source_records,
        "code_sha256": {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in code_paths},
        "archive_sha256": digest(archive_path.read_bytes()), "recorded_commit_checks": commit_checks,
        "native_scaling_evidence": "Code/configuration interpretation: the released runner defaults to native; native apply_attention_scaling returns without modifying the backbone. Original retained JSON has no attention-scaling field. Later alpha-controller cohorts are excluded.",
        "provenance_scope": "Archived JSON metrics and norms are checksum-verified; released code explains the implemented operations. The recorded Git HEAD does not contain the runner path, so it is not an exact per-run source snapshot. No historical checkpoint was reloaded. Matching released YAMLs identify the workflow; archived JSON, not present-day YAML defaults, supplies reported optimizer settings.",
        "activity_scope": "Initial measured norm changes are computed from the original before/after records. Raw norm > initial gamma is also reported, separately; finite precision can make the two counts differ. Checkpoint weight_stats_final values are effective norms, not uncapped raw norms.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUTPUT / "initial_matrix_norms.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrices[0]))
        writer.writeheader()
        writer.writerows(matrices)

    print(json.dumps({"status": "passed", "original_table_rows": len(rows),
                      "records_verified": len(source_records), "matrix_records": len(matrices),
                      "output": str(OUTPUT.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
