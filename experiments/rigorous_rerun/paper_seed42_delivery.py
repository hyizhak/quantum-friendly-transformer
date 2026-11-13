"""Evaluate the completed seed-42 multilayer classification cohort.

Run with PYTHONPATH set to the frozen study source and its src directory.
The training protocol and validation selection rule remain frozen. This watcher
only evaluates the completed seed-42 multilayer FN job, then exports a new table.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def alive(pid):
    try:
        os.kill(pid, 0)
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except (ProcessLookupError, FileNotFoundError):
        return False


def main(args):
    root, repo = args.root.resolve(), args.repository.resolve()
    folder = root / "paper_seed42"
    status = folder / "delivery_status.json"
    plan = json.loads((folder / "minimum_paper_plan.json").read_text())
    pid = plan["workers"][0]
    output = folder / "final_eval"
    output.mkdir(exist_ok=True)
    write_json(status, {"phase": "waiting_for_200_epochs", "training_pid": pid,
                        "watcher_pid": os.getpid(), "seed": 42})
    while not (folder / "fn_training_complete.json").exists():
        if not alive(pid):
            raise RuntimeError("Training exited before its completion record; no partial result will be published")
        time.sleep(10)
    # Release the trainer's CUDA context before loading the selected checkpoint.
    while alive(pid):
        time.sleep(2)

    import numpy as np
    import torch
    from experiments.rigorous_rerun.run import (
        build_model, configure, evaluate, norm_audit, settings, verify_study,
    )
    configure()
    study = json.loads((root / "study.json").read_text())
    verify_study(study, root)
    job = next(j for j in study["jobs"] if j["id"] == "genomic/42/multilayer/fn")
    run = root / "runs" / job["id"]
    selected = json.loads((run / "selected.json").read_text())
    config = settings(study["protocol"], "genomic", "multilayer")
    assert config["epochs"] == selected["trained_epochs"] == 200
    assert config["metric"] == "accuracy" and not selected["stopped_early"]
    assert selected["study_sha256"] == study["study_sha256"]
    assert sha(run / "best.pt") == selected["checkpoint_sha256"]
    history = json.loads((run / "history.json").read_text())
    winner = max((r for r in history if r["epoch"] > 0), key=lambda r: r["validation"]["accuracy"])
    assert winner["epoch"] == selected["best_epoch"]
    lock_path = output / "selection_lock.json"
    lock = {"selection": selected, "study_sha256": study["study_sha256"],
            "scope": "User-requested single seed, paper multilayer FN row only; earlier multi-seed scope superseded.",
            "test_policy": "Evaluate once after the full 200-epoch protocol and validation-only checkpoint selection.",
            "evaluator_sha256": sha(__file__)}
    if lock_path.exists():
        assert json.loads(lock_path.read_text()) == lock
    else:
        write_json(lock_path, lock)
    write_json(status, {"phase": "evaluating_locked_checkpoint", "selected_epoch": selected["best_epoch"]})
    record_path = output / "multilayer_fn.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        assert record["checkpoint_sha256"] == selected["checkpoint_sha256"]
        assert record["selection_lock_sha256"] == sha(lock_path)
        assert record["predictions_sha256"] == sha(output / "test_predictions.npz")
    else:
        model, coverage = build_model(study, root, job)
        state = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        model.cuda()
        model.load_state_dict(state["model"], strict=True)
        validation = torch.load(root / "data/genomic/validation.pt", weights_only=True)
        metrics, _, _ = evaluate(model, validation, job, config["eval_batch_size"])
        assert metrics == selected["validation"], (metrics, selected["validation"])
        test = torch.load(root / "data/genomic/test.pt", weights_only=True)
        metrics, predictions, logits = evaluate(model, test, job, config["eval_batch_size"], return_logits=True)
        actual, predicted = test["labels"].numpy(), predictions.numpy()
        assert len(actual) == 9034 and logits.shape == (9034, 2) and logits.dtype == torch.float32
        assert torch.isfinite(logits).all() and np.array_equal(logits.argmax(-1).numpy(), predicted)
        tp = int(((actual == 1) & (predicted == 1)).sum())
        tn = int(((actual == 0) & (predicted == 0)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())
        assert abs(metrics["accuracy"] - (tp + tn) / len(actual)) < 1e-12
        assert abs(metrics["f1"] - 2 * tp / (2 * tp + fp + fn)) < 1e-12
        norms = norm_audit(model)
        assert len(norms) == 72 and all(abs(row["frobenius"] - 1) < 2e-5 for row in norms)
        prediction_path = output / "test_predictions.npz"
        np.savez_compressed(prediction_path, labels=actual, predictions=predicted, logits=logits.numpy())
        record = {"job": job, "seed": 42, "trained_epochs": 200, "selected_epoch": state["epoch"],
                  "validation": selected["validation"], "test": metrics, "norms": norms,
                  "confusion_counts": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
                  "initialization": coverage, "checkpoint_sha256": selected["checkpoint_sha256"],
                  "study_sha256": study["study_sha256"], "selection_lock_sha256": sha(lock_path),
                  "predictions_sha256": sha(prediction_path), "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        write_json(record_path, record)
    singles = json.loads((repo / "artifacts/rigorous_rerun/single_seed_42/summary.json").read_text())["rows"]
    combined = {"seed": 42, "test_examples": 9034, "single_layer": singles, "multilayer_fn": record}
    write_json(output / "paper_summary.json", combined)
    lines = ["model,seed,trained_epochs,selected_epoch,test_accuracy,test_f1"]
    for row in singles:
        lines.append(f"single_{row['variant']},42,100,{row['selected_epoch']},{row['test']['accuracy']},{row['test']['f1']}")
    lines.append(f"multilayer_fn,42,200,{record['selected_epoch']},{record['test']['accuracy']},{record['test']['f1']}")
    (output / "paper_metrics.csv").write_text("\n".join(lines) + "\n")
    shutil.copy2(__file__, output / "evaluator.py")
    exported = repo / "artifacts/rigorous_rerun/paper_seed42"
    shutil.copytree(output, exported, dirs_exist_ok=True)
    write_json(status, {"phase": "complete", "test": record["test"], "selected_epoch": record["selected_epoch"],
                        "report": str(exported / "paper_summary.json")})
    print(json.dumps({"status": "complete", "test": record["test"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        write_json(args.root / "paper_seed42/delivery_status.json", {"phase": "failed", "error": traceback.format_exc()})
        raise
