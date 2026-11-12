"""Repeat the historically successful 3M bounded case with the new protocol."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/configs/recovery/recovery_positive_control.json")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--mode", choices=["bounded", "uniform"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.single_run:
        from experiments.runners.recovery_extension import run
        assert args.mode is not None and args.seed is not None and args.device is not None
        run(argparse.Namespace(config=args.config, assets=args.assets, output=args.output,
                               model="3m", mode=args.mode, seed=args.seed, device=args.device,
                               compile=True, steps=None, eval_every=None, smoke=False))
        path = args.output / "manifest.json"
        metadata = json.loads(path.read_text())
        metadata["source_sha256"][str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        path.write_text(json.dumps(metadata, indent=2) + "\n")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    config = json.loads(args.config.read_text())

    def worker(mode, device):
        for seed in config["seeds"]:
            name = f"3m_{mode}_seed{seed}"
            output = args.output / name
            if (output / "manifest.json").exists():
                continue
            if output.exists():
                raise RuntimeError(f"Incomplete output: {output}")
            command = [sys.executable, str(Path(__file__).resolve()), "--single-run",
                       "--config", str(args.config), "--assets", str(args.assets),
                       "--output", str(output), "--mode", mode, "--seed", str(seed), "--device", device]
            print(f"Starting {name}", flush=True)
            with (args.output / "logs" / f"{name}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((output / "result.json").read_text())
            print(f"Completed {name}: accuracy={result['evaluation']['accuracy']:.6f}, threshold_met={result['meets_historical_threshold']}", flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, "bounded", "cuda:0"), pool.submit(worker, "uniform", "cuda:1")]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
