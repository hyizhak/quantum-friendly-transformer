"""Evaluate the explicitly requested completed seed-42 non-TATA cohort.

Execute this file with PYTHONPATH pointing at the frozen study source and src.
The original study stays immutable; the user-requested scope/test-barrier
amendment and all three selected checkpoints are locked before evaluation.
"""

import argparse
import ast
import gc
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset

from experiments.rigorous_rerun.run import (
    build_model,
    configure,
    evaluate,
    file_sha,
    norm_audit,
    score,
    verify_study,
    write_json,
)


def binary_metrics(labels, predictions):
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return {"accuracy": float((labels == predictions).mean()),
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    root, repo = args.root.resolve(), args.repository.resolve()
    study = json.loads((root / "study.json").read_text())
    request = json.loads((root / "single_seed_request.json").read_text())
    if request["delivery_scope"] != {"dataset": "genomic", "family": "single", "seed": 42, "variants": ["vanilla", "sn", "fn"]}:
        raise RuntimeError("Delivery scope does not match the user request")
    configure()
    verify_study(study, root)
    output = root / "single_seed_42"
    output.mkdir(exist_ok=True)
    jobs = [j for j in study["jobs"] if j["dataset"] == "genomic" and j["family"] == "single" and j["seed"] == 42]
    if [j["variant"] for j in jobs] != ["vanilla", "sn", "fn"]:
        raise AssertionError("Incomplete three-model cohort")
    selections = {}
    for job in jobs:
        folder = root / "runs" / job["id"]
        record = json.loads((folder / "selected.json").read_text())
        if record["study_sha256"] != study["study_sha256"] or record["trained_epochs"] != 100:
            raise AssertionError("Each model must complete the prespecified 100 epochs")
        if record["checkpoint_sha256"] != file_sha(folder / "best.pt"):
            raise AssertionError("Selected checkpoint checksum mismatch")
        selections[job["id"]] = record
    n_test = study["data"]["genomic"]["splits"]["test"]["rows"]
    # Reconstruct the old script's held-out half for an explicitly secondary comparison.
    legacy_indices = np.array(Dataset.from_dict({"row_index": list(range(n_test))}).train_test_split(
        test_size=0.5, seed=42)["test"]["row_index"], dtype=np.int64)
    np.save(output / "legacy_rule_test_indices.npy", legacy_indices)
    old_text = (repo / "results/genomic_single_layer.txt").read_text()
    labels = {"vanilla": "vanilla", "sn_model": "sn", "fn_model": "fn"}
    historical = {labels[name]: ast.literal_eval(values) for name, values in
                  re.findall(r"^(vanilla|sn_model|fn_model) final test metrics: (\{[^\n]+\})$", old_text, re.MULTILINE)}
    if set(historical) != {"vanilla", "sn", "fn"}:
        raise AssertionError("Historical log has missing/duplicate model rows")
    cohort = {"locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "study_sha256": study["study_sha256"], "request": request, "selections": selections,
              "primary_evaluation": "Complete official test split, 9034 examples; fixed training-derived validation split selects checkpoints by F1.",
              "secondary_evaluation": "Subset reconstructed from the old script's train_test_split(test_size=0.5, seed=42); 4517 examples. Historical dataset revision/split identities were not archived, so exact identity with the historical run is not certified.",
              "legacy_indices_sha256": file_sha(output / "legacy_rule_test_indices.npy"),
              "historical_log_sha256": file_sha(repo / "results/genomic_single_layer.txt"),
              "evaluator_sha256": file_sha(__file__), "selection_rule_changed": False,
              "single_seed_limitation": "No across-seed uncertainty or robustness claim; seed 42 was fixed by the original recipe, not selected by test score."}
    lock = output / "cohort.json"
    if lock.exists():
        old = json.loads(lock.read_text())
        if old["selections"] != selections or old["evaluator_sha256"] != cohort["evaluator_sha256"]:
            raise RuntimeError("Cannot change the cohort after test access")
    else:
        write_json(lock, cohort)
    validation = torch.load(root / "data/genomic/validation.pt", weights_only=True)
    test = torch.load(root / "data/genomic/test.pt", weights_only=True)
    rows = []
    for job in jobs:
        variant = job["variant"]
        model, _ = build_model(study, root, job)
        state = torch.load(root / "runs" / job["id"] / "best.pt", map_location="cpu", weights_only=True)
        model.cuda()
        model.load_state_dict(state["model"], strict=True)
        val_metrics, _, _ = evaluate(model, validation, job, 64)
        if val_metrics != selections[job["id"]]["validation"]:
            raise AssertionError("Reloaded validation metrics differ from the frozen selection")
        metrics, predictions, logits = evaluate(model, test, job, 64, return_logits=True)
        actual, predicted = test["labels"].numpy(), predictions.numpy()
        for key, value in binary_metrics(actual, predicted).items():
            if not math.isclose(metrics[key], value, rel_tol=0, abs_tol=1e-12):
                raise AssertionError("Independent confusion-count metric verification failed")
        auxiliary = score(actual[legacy_indices], predicted[legacy_indices], False)
        path = output / f"{variant}_test_predictions.npz"
        np.savez_compressed(path, labels=actual, predictions=predicted, logits=logits.numpy())
        row = {"variant": variant, "seed": 42, "trained_epochs": 100, "selected_epoch": state["epoch"],
               "validation": val_metrics, "test": metrics, "legacy_rule_subset": auxiliary,
               "historical_log_test": historical[variant], "norms": norm_audit(model),
               "checkpoint_sha256": selections[job["id"]]["checkpoint_sha256"],
               "predictions_sha256": file_sha(path)}
        rows.append(row)
        write_json(output / f"{variant}.json", row)
        print(json.dumps(row), flush=True)
        del model, state, predictions, logits
        gc.collect()
        torch.cuda.empty_cache()
    report = {"study_sha256": study["study_sha256"], "cohort_sha256": file_sha(lock),
              "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "rows": rows, "limitations": [cohort["single_seed_limitation"],
              "Training/validation split, precision and repaired model implementation differ from the old experiment; numerical differences are not attributable solely to code fixes.",
              cohort["secondary_evaluation"]]}
    write_json(output / "summary.json", report)
    lines = ["# Corrected non-TATA single-layer results, seed 42", "",
             "All models completed 100 epochs. Checkpoints were locked using validation F1 before test evaluation.", "",
             "| Model | Selected epoch | Full test accuracy | Full test F1 |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['variant']} | {row['selected_epoch']} | {row['test']['accuracy']*100:.4f}% | {row['test']['f1']*100:.4f}% |")
    lines += ["", "Primary test set: 9,034 examples. One seed; no across-seed mean or standard deviation.", "",
              "| Model | Historical log accuracy | New accuracy on reconstructed old-rule subset |", "|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['variant']} | {row['historical_log_test']['accuracy']*100:.4f}% | {row['legacy_rule_subset']['accuracy']*100:.4f}% |")
    lines += ["", *report["limitations"], "", "The requested multi-seed queue was stopped; all previous checkpoints and completed runs are retained."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    shutil.copy2(__file__, output / "evaluator.py")
    exported = repo / "artifacts/rigorous_rerun/single_seed_42"
    shutil.copytree(output, exported, dirs_exist_ok=True)
    write_json(root / "supervisor_status.json", {"phase": "single_seed_delivery_complete",
               "study_sha256": study["study_sha256"], "report": str(exported / "summary.md"),
               "multi_seed_queue_running": False, "automatic_multiseed_resume": False})


if __name__ == "__main__":
    main()
