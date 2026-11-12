"""Run the predeclared recovery extension on two GPUs, preserving all results."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=(Path(__file__).resolve().parents[1] / 'configs/recovery/recovery_extension.json'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / "logs"
    logs.mkdir(exist_ok=True)
    config = json.loads(args.config.read_text())
    seeds = config["seeds"]
    gpu0 = [(model, "bounded", seed) for seed in seeds for model in ["1m", "8m"]]
    gpu0 += [("1m", "uniform", seed) for seed in seeds]
    gpu1 = [("33m", "bounded", seed) for seed in seeds]
    gpu1 += [(model, "uniform", seed) for seed in seeds for model in ["33m", "8m"]]

    def worker(device, jobs):
        for model, mode, seed in jobs:
            name = f"{model}_{mode}_seed{seed}"
            output = args.output / name
            if (output / "manifest.json").exists():
                result = json.loads((output / "result.json").read_text())
                assert not result["smoke"]
                print(f"Already complete: {name}", flush=True)
                continue
            if output.exists():
                raise RuntimeError(f"Incomplete prior output; inspect before retrying: {output}")
            command = [sys.executable, str(Path(__file__).with_name('recovery_extension.py')),
                       "--config", str(args.config), "--assets", str(args.assets),
                       "--model", model, "--mode", mode, "--seed", str(seed),
                       "--device", device, "--compile", "--output", str(output)]
            print(f"Starting {name} on {device}", flush=True)
            with (logs / f"{name}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((output / "result.json").read_text())
            print(f"Completed {name}: accuracy={result['evaluation']['accuracy']:.6f}, threshold_met={result['meets_historical_threshold']}", flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, "cuda:0", gpu0), pool.submit(worker, "cuda:1", gpu1)]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
