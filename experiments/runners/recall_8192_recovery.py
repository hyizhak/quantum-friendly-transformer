"""Validation-driven cap24 training extensions and constrained fine-tuning."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from experiments.lib.experiment import load_config
from experiments.runners import recall_8192 as base
from experiments.lib.long_recall import matrix_records

SEARCH_PROTOCOL = ROOT / "experiments/configs/recall/RECALL_8192_RECOVERY_SEARCH.txt"
FROZEN_MAKE_MODEL = base.routing.core.make_model


def initial_model(config, variant, device):
    model = FROZEN_MAKE_MODEL(config, variant, device)
    if config.get("warm_start_root"):
        seed = torch.initial_seed()
        assert seed in config["seeds"]
        parent = Path(config["warm_start_root"]) / f"native_sqrt_d_seed{seed}"
        result = json.loads((parent / "result.json").read_text())
        assert hashlib.sha256((parent / "final.pth").read_bytes()).hexdigest() == result["checkpoint_sha256"]
        saved = torch.load(parent / "final.pth", map_location=device, weights_only=True)
        assert saved["config"]["cap"] == 64 and config["cap"] == 24
        model.load_state_dict(saved["state_dict"])
        # max_gamma is a constructor attribute, not a state_dict value. Loading
        # the cap64 raw parameters and g does not load the old maximum cap.
        assert model.cap == 24
        for module in model.modules():
            if hasattr(module, "max_gamma"):
                assert module.max_gamma == 24
        assert max(r["effective_frobenius"] for r in matrix_records(model)) <= 24 * (1 + 1e-7)
    return model


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        base.main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint-root', type=Path, help='Root for relative parent-checkpoint paths in the recipe')
    parser.add_argument("--config", type=Path, required=True)
    args, _ = parser.parse_known_args()
    config = load_config(args.config, getattr(args, "checkpoint_root", None))
    parents = {}
    if config.get("warm_start_root"):
        for seed in config["seeds"]:
            parent = Path(config["warm_start_root"]) / f"native_sqrt_d_seed{seed}"
            provenance = json.loads((parent / "manifest.json").read_text())
            for name, expected in provenance["source_sha256"].items():
                assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
            parents[str(seed)] = {
                "run": str(parent.resolve()),
                "checkpoint_sha256": hashlib.sha256((parent / "final.pth").read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256((parent / "manifest.json").read_bytes()).hexdigest(),
                "additional_parent_training_updates": 3000,
            }
    frozen_manifest = base.routing.manifest
    def search_manifest(run_config, sources, **extra):
        return frozen_manifest(run_config, sources + [Path(__file__), SEARCH_PROTOCOL],
                               parent_runs=parents,
                               initialization="cap64_parent_with_maximum24" if parents else "from_scratch",
                               **extra)
    base.routing.manifest = search_manifest
    base.routing.core.make_model = initial_model
    base.main()


if __name__ == "__main__":
    main()
