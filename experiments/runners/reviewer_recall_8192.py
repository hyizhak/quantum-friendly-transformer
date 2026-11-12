"""Run a disclosed long-recall recipe through N=8192 using frozen model code."""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from experiments.runners import reviewer_long_recall_routing as routing
from experiments.lib.reviewer import save_csv, save_json
from experiments.lib.long_recall import matrix_records

PROTOCOL = ROOT / "experiments/configs/reviewer/RECALL_8192_CAP24_PROTOCOL.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["native_sqrt_d", "uniform"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert config["curriculum"][-1]["N"] == 8192
    assert config["evaluation_role"] == "validation"
    assert config["cap"] in (24, 64)
    assert args.variant in config["variants"]
    if args.smoke:
        config = copy.deepcopy(config)
        config.update(seeds=[11], steps=3, log_every=1, test_pairs=8,
                      test_batch_pairs=4, test_lengths=[8192],
                      curriculum=[{"through_step": 3, "N": 8192}])
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config
    else:
        # Each worker writes the same complete immutable configuration.
        try:
            with config_path.open("x") as stream:
                stream.write(json.dumps(config, indent=2) + "\n")
        except FileExistsError:
            assert json.loads(config_path.read_text()) == config
    plan_path = args.output / f"plan_{args.variant}.json"
    if plan_path.exists():
        raise FileExistsError(f"Refusing to overwrite {plan_path}")
    plan = [{"seed": seed, "variant": args.variant, "status": "planned"}
            for seed in config["seeds"]]
    save_json(plan_path, plan)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    frozen_manifest = routing.manifest
    def manifest_with_protocol(run_config, sources, **extra):
        return frozen_manifest(run_config, sources + [Path(__file__), PROTOCOL, args.config],
                               evaluation_role="validation", final_test_used=False,
                               recipe=config["protocol"], **extra)
    routing.manifest = manifest_with_protocol

    for item in plan:
        item["status"] = "running"
        save_json(plan_path, plan)
        torch.manual_seed(item["seed"])
        initial_model = routing.core.make_model(config, args.variant, device)
        save_csv(args.output / f"initial_{args.variant}_seed{item['seed']}.csv",
                 matrix_records(initial_model))
        del initial_model
        routing.run_one(config, args.output / f"{args.variant}_seed{item['seed']}",
                        item["seed"], args.variant, device, args.smoke)
        item["status"] = "completed"
        save_json(plan_path, plan)


if __name__ == "__main__":
    main()
