"""Test each FN1 variant's validation-selected checkpoint, as requested."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def worker(root, worker_id):
    import numpy as np
    import torch

    output = root / "all_variants_test"
    lock_path = output / "selection_lock.json"
    lock = json.loads(lock_path.read_text())
    assert sha(root / "plan.json") == lock["plan_sha256"]
    spec = importlib.util.spec_from_file_location("frozen_fn1", root / "unit_fn_search.py")
    search = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(search)
    search.base.configure()
    plan, original, study = search.load_plan(root)
    validation = torch.load(original / "data/genomic/validation.pt", weights_only=True)
    test_path = original / "data/genomic/test.pt"
    assert sha(test_path) == plan["data_sha256"]["test"]
    test = torch.load(test_path, weights_only=True)
    labels = test["labels"].numpy()
    assert len(labels) == 9034
    for selected in lock["candidates"]:
        trial = selected["trial"]
        if trial["worker"] != worker_id:
            continue
        folder = output / trial["name"]
        folder.mkdir(exist_ok=True)
        checkpoint = root / trial["name"] / "best.pt"
        assert sha(checkpoint) == selected["checkpoint_sha256"]
        report_path = folder / "result.json"
        if report_path.exists():
            saved = json.loads(report_path.read_text())
            assert saved["selection_lock_sha256"] == sha(lock_path)
            assert saved["checkpoint_sha256"] == sha(checkpoint)
            assert saved["predictions_sha256"] == sha(folder / "test_predictions.npz")
            continue
        model = search.build(plan, original, study, trial).cuda()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        del state
        metrics, predictions, _ = search.base.evaluate(
            model, validation, search.JOB, trial["eval_batch_size"])
        with np.load(root / trial["name"] / "validation_predictions.npz") as saved:
            assert np.array_equal(predictions.numpy(), saved["predictions"])
        assert metrics["accuracy"] == selected["validation"]["accuracy"]
        norms = search.audit(model)
        metrics, predictions, logits = search.base.evaluate(
            model, test, search.JOB, trial["eval_batch_size"], return_logits=True)
        predictions = predictions.numpy()
        assert logits.shape == (9034, 2) and torch.isfinite(logits).all()
        counts = {
            "tp": int(((labels == 1) & (predictions == 1)).sum()),
            "tn": int(((labels == 0) & (predictions == 0)).sum()),
            "fp": int(((labels == 0) & (predictions == 1)).sum()),
            "fn": int(((labels == 1) & (predictions == 0)).sum()),
        }
        assert sum(counts.values()) == 9034
        assert metrics["accuracy"] == (counts["tp"] + counts["tn"]) / 9034
        assert abs(metrics["f1"] - 2 * counts["tp"] /
                   (2 * counts["tp"] + counts["fp"] + counts["fn"])) < 1e-12
        np.savez_compressed(folder / "test_predictions.npz", labels=labels,
                            predictions=predictions, logits=logits.numpy())
        report = {
            "trial": trial, "selected_epoch": selected["selected_epoch"],
            "trained_epochs": selected["trained_epochs"],
            "validation": selected["validation"], "test": metrics,
            "confusion_counts": counts, "norms": norms,
            "checkpoint_sha256": sha(checkpoint),
            "selection_lock_sha256": sha(lock_path),
            "predictions_sha256": sha(folder / "test_predictions.npz"),
            "prior_test_disclosure": plan["prior_test_disclosure"],
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write(report_path, report)
        print(json.dumps({"trial": trial["name"], "test": metrics}), flush=True)
        del model
        torch.cuda.empty_cache()


def deliver(root):
    plan = json.loads((root / "plan.json").read_text())
    output = root / "all_variants_test"
    output.mkdir(exist_ok=True)
    with (output / "delivery.lock").open("a") as delivery_lock:
        fcntl.flock(delivery_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(output / "policy.json", {
            "user_request": "四个都选一个最高的test啊",
            "interpretation": "For each of the four variants, choose its highest-validation-accuracy checkpoint and evaluate it once on the full official test set; report all four.",
            "supersedes": "The original plan's single-variant final test and the intermediate single-winner delivery policy. Training and checkpoint selection remain unchanged.",
            "plan_sha256": sha(root / "plan.json"), "script_sha256": sha(Path(__file__)),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        deadline = time.monotonic() + 3600
        while not all((root / f"worker_{worker_id}_complete.json").exists()
                      for worker_id in {t["worker"] for t in plan["trials"]}):
            if time.monotonic() > deadline:
                raise TimeoutError("Training workers did not finish within one hour")
            time.sleep(5)
        candidates = [json.loads((root / t["name"] / "selected.json").read_text())
                      for t in plan["trials"]]
        for selected in candidates:
            assert sha(root / selected["trial"]["name"] / "best.pt") == selected["checkpoint_sha256"]
        lock = {
            "plan_sha256": sha(root / "plan.json"), "candidates": candidates,
            "rule": "Best validation accuracy within each variant; first checkpoint wins ties. Report all four test results.",
            "validation_selected_variant": max(candidates, key=lambda x: x["validation"]["accuracy"])["trial"]["name"],
            "prior_test_disclosure": plan["prior_test_disclosure"],
        }
        lock_path = output / "selection_lock.json"
        if lock_path.exists():
            assert json.loads(lock_path.read_text()) == lock
        else:
            write(lock_path, lock)
        original = Path(plan["original_study"])
        env = os.environ.copy()
        env.update(HF_HOME="/data/hyizhan/huggingface", HF_HUB_OFFLINE="1",
                   TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   PYTHONHASHSEED="0", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   PYTHONPATH=f"{original}/source:{original}/source/src")
        snapshot = output / "test_unit_fn_variants.py"
        if not snapshot.exists():
            shutil.copy2(Path(__file__), snapshot)
        assert sha(snapshot) == sha(Path(__file__))
        processes = []
        for worker_id in sorted({t["worker"] for t in plan["trials"]}):
            with (output / f"worker_{worker_id}.log").open("a") as log:
                process = subprocess.Popen(
                    [sys.executable, "-u", str(snapshot), "worker", "--root", str(root), "--worker", str(worker_id)],
                    cwd=original / "source", env={**env, "CUDA_VISIBLE_DEVICES": str(worker_id)},
                    stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
        codes = [process.wait() for process in processes]
        assert codes == [0] * len(processes), f"Test worker exit codes: {codes}"
        reports = [json.loads((output / t["name"] / "result.json").read_text())
                   for t in plan["trials"]]
        for report in reports:
            assert report["selection_lock_sha256"] == sha(lock_path)
            assert report["predictions_sha256"] == sha(output / report["trial"]["name"] / "test_predictions.npz")
        summary = {"validation_selected_variant": lock["validation_selected_variant"],
                   "seed": 42, "test_examples": 9034, "reports": reports,
                   "prior_test_disclosure": plan["prior_test_disclosure"]}
        write(output / "summary.json", summary)
        repository = Path(plan["repository"])
        exported = repository / "artifacts/rigorous_rerun/unit_fn_all_tests"
        shutil.copytree(output, exported, dirs_exist_ok=True)
        shutil.copy2(root / "plan.json", exported / "plan.json")
        for trial in plan["trials"]:
            shutil.copy2(root / trial["name"] / "history.json", exported / trial["name"] / "history.json")
        write(output / "complete.json", {"summary": str(exported / "summary.json"),
                                         "test_variants": len(reports)})
        shutil.copy2(output / "complete.json", exported / "complete.json")
        print(json.dumps({"complete": True, "results": [
            {"trial": r["trial"]["name"], "test": r["test"]} for r in reports]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["deliver", "worker"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    if args.action == "deliver":
        deliver(args.root.resolve())
    else:
        worker(args.root.resolve(), args.worker)
