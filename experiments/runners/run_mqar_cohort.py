"""Run all fixed MQAR seeds for one attention variant on one device."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["native", "uniform"], required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--runner", type=Path, default=ROOT / "experiments/runners/reviewer_mqar.py")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/configs/reviewer/mqar.json")
    parser.add_argument("--assets", type=Path, default=ROOT / "results/reviewer/mqar_data")
    parser.add_argument("--output", type=Path, default=ROOT / "results/reviewer/mqar")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in config["seeds"]:
        name = f"{args.variant}_seed{seed}"
        if (args.output / name).exists() or (args.output / f"{name}.console.log").exists():
            raise FileExistsError(name)
    environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS=str(config["threads"]), MKL_NUM_THREADS=str(config["threads"]))
    for seed in config["seeds"]:
        name = f"{args.variant}_seed{seed}"
        command = [sys.executable, str(args.runner), "train",
                   "--config", str(args.config), "--assets", str(args.assets),
                   "--output", str(args.output / name), "--variant", args.variant,
                   "--seed", str(seed), "--device", args.device]
        print(f"START {name} {args.device}", flush=True)
        with (args.output / f"{name}.console.log").open("x") as stream:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f"COMPLETE {name}", flush=True)


if __name__ == "__main__":
    main()
