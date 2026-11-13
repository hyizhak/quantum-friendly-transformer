"""Finish the prespecified FN1 cohort and report its validation-selected test result."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(root):
    plan = json.loads((root / "plan.json").read_text())
    processes = json.loads((root / "processes.json").read_text())
    original = Path(plan["original_study"])
    repository = Path(plan["repository"])
    runner = root / "unit_fn_search.py"
    assert sha256(runner) == plan["runner_sha256"]

    # This lock protects against duplicate delivery watchers. The earlier
    # supervisor is allowed to finish before this watcher invokes finalize.
    with (root / "delivery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        policy = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "plan_sha256": sha256(root / "plan.json"),
            "delivery_script_sha256": sha256(Path(__file__)),
            "rule": "Complete all four existing trials, select the maximum validation accuracy, then evaluate that checkpoint on test once, regardless of the validation score.",
            "changes": "Remove the earlier launcher's 0.925 threshold for automatic delivery; training, candidates, selection rule, and frozen runner are unchanged.",
        }
        policy_path = root / "test_delivery_policy.json"
        if not policy_path.exists():
            policy_path.write_text(json.dumps(policy, indent=2) + "\n")

        deadline = time.monotonic() + 3600
        command_path = Path(f"/proc/{processes['supervisor_pid']}/cmdline")
        while command_path.exists():
            try:
                command = command_path.read_bytes()
            except FileNotFoundError:
                break
            if b"qft_launch_fn1_search.py" not in command:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Existing supervisor did not finish within one hour")
            time.sleep(5)

        if not (root / "complete.json").exists():
            for worker in {trial["worker"] for trial in plan["trials"]}:
                assert (root / f"worker_{worker}_complete.json").exists(), "A worker did not complete successfully"
            candidates = [json.loads((root / trial["name"] / "selected.json").read_text())
                          for trial in plan["trials"]]
            chosen = max(candidates, key=lambda x: x["validation"]["accuracy"])
            env = os.environ.copy()
            env.update(
                CUDA_VISIBLE_DEVICES=str(chosen["trial"]["worker"]),
                HF_HOME="/data/hyizhan/huggingface", HF_HUB_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                PYTHONHASHSEED="0", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                PYTHONPATH=f"{original}/source:{original}/source/src",
            )
            subprocess.run(
                [sys.executable, "-u", str(runner), "finalize", "--root", str(root)],
                cwd=original / "source", env=env, check=True,
            )

        report = json.loads((root / "final/result.json").read_text())
        assert report["predictions_sha256"] == sha256(root / "final/test_predictions.npz")
        assert report["selection_lock_sha256"] == sha256(root / "selection_lock.json")
        exported = repository / "artifacts/rigorous_rerun/unit_fn_search"
        (exported / "test_delivery_policy.json").write_text(policy_path.read_text())
        print((root / "complete.json").read_text(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    main(args.root.resolve())
