"""Supervise both frozen workers, then unlock test evaluation and aggregate."""

import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def supervise(root):
    (root / "logs").mkdir(exist_ok=True)
    for phase in ("queue", "finalize"):
        with contextlib.ExitStack() as stack:
            children = []
            for worker in range(2):
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(worker), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                       "PYTHONHASHSEED": "0", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
                       "HF_HOME": "/data/hyizhan/huggingface", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                       "PYTHONPATH": f"{root}/source:{root}/source/src"}
                stream = stack.enter_context(open(root / "logs" / f"{phase}_worker{worker}.log", "a", buffering=1))
                process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.rigorous_rerun.run", phase,
                                            "--root", str(root), "--worker", str(worker), "--workers", "2"],
                                           cwd=root / "source", env=env, stdout=stream, stderr=subprocess.STDOUT)
                children.append(process)
            (root / "supervisor_status.json").write_text(json.dumps({"phase": phase, "supervisor_pid": os.getpid(),
                       "workers": [child.pid for child in children], "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
            statuses = [child.wait() for child in children]
        if any(statuses):
            raise RuntimeError(f"{phase} workers exited with {statuses}; inspect logs before resuming")
    env = {**os.environ, "PYTHONPATH": f"{root}/source:{root}/source/src"}
    subprocess.run([sys.executable, "-m", "experiments.rigorous_rerun.run", "summarize", "--root", str(root)],
                   cwd=root / "source", env=env, check=True)
    (root / "supervisor_status.json").write_text(json.dumps({"phase": "complete", "supervisor_pid": os.getpid()}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    with open(root / "supervisor.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            supervise(root)
        except Exception as error:
            (root / "supervisor_status.json").write_text(json.dumps({"phase": "failed", "supervisor_pid": os.getpid(), "error": repr(error)}))
            raise


if __name__ == "__main__":
    main()
