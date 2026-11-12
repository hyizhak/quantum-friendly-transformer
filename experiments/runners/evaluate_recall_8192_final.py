"""Evaluate a validation-selected cap24 recipe on the reserved final test."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from experiments.lib.experiment import manifest, save_json
from experiments.runners import long_recall as core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    assert selection["selection_split"] == "validation"
    assert selection["final_test_used_for_selection"] is False
    assert selection["all_recipes_complete"]
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "selection.json", selection)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    sources = [Path(__file__), Path(core.__file__), ROOT / "experiments/lib/long_recall.py",
               ROOT / "experiments/lib/experiment.py",
               ROOT / "src/quantum_friendly_transformer/norm_transformer/frobenius_cap.py",
               ROOT / "src/quantum_friendly_transformer/norm_transformer/normalization.py",
               ROOT / "experiments/configs/recall/RECALL_8192_CAP24_PROTOCOL.txt",
               ROOT / "experiments/configs/recall/RECALL_8192_RECOVERY_SEARCH.txt", args.selection]
    groups = [("selected_cap24", selection["selected_cap24_root"], "native_sqrt_d"),
              ("reference_cap64", selection["reference_cap64_root"], "native_sqrt_d"),
              ("uniform_cap24", selection["uniform_cap24_root"], "uniform")]
    for label, directory, variant in groups:
        group = args.output / label
        group.mkdir()
        for seed in (11, 22, 33):
            parent = Path(directory) / f"{variant}_seed{seed}"
            parent_result = json.loads((parent / "result.json").read_text())
            parent_manifest = json.loads((parent / "manifest.json").read_text())
            for path, expected in parent_manifest["source_sha256"].items():
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
            checksum = hashlib.sha256((parent / "final.pth").read_bytes()).hexdigest()
            assert checksum == parent_result["checkpoint_sha256"]
            checkpoint = torch.load(parent / "final.pth", map_location=device, weights_only=True)
            training_config = checkpoint["config"]
            config = {**training_config, "test_seed": 7041937, "test_pairs": 2048,
                      "evaluation_role": "final_test"}
            output = group / f"{variant}_seed{seed}"
            output.mkdir()
            save_json(output / "config.json", config)
            provenance = manifest(config, sources, seed=seed, variant=variant,
                parent_training_run=str(parent.resolve()), training_config=training_config,
                training_performed=False, evaluation_role="final_test", device=str(device),
                evaluation_dtype="float64", checkpoint_sha256=checksum,
                selection_manifest_sha256=hashlib.sha256(args.selection.read_bytes()).hexdigest(),
                parent_manifest_sha256=hashlib.sha256((parent / "manifest.json").read_bytes()).hexdigest())
            save_json(output / "running_manifest.json", provenance)
            shutil.copyfile(parent / "final.pth", output / "final.pth")
            model = core.make_model(training_config, variant, device)
            model.load_state_dict(checkpoint["state_dict"])
            started = time.perf_counter()
            metrics = core.evaluate(model, config, output, device)
            evaluations = []
            for n in config["test_lengths"]:
                batches = [r for r in metrics if r["N"] == n]
                count = sum(r["examples"] for r in batches)
                evaluations.append({"N": n, "examples": count,
                    **{prefix+"accuracy": sum(r[prefix+"correct"] for r in batches) / count
                       for prefix in ["", "uniform_", "shuffled_query_"] +
                       [f"noise_{e:g}_" for e in config["attention_noise_l2"]]}})
            result = {**parent_result, "evaluation": evaluations, "reevaluation_only": True,
                      "evaluation_role": "final_test", "parent_training_run": str(parent.resolve()),
                      "evaluation_seconds": time.perf_counter() - started}
            save_json(output / "result.json", result)
            save_json(output / "manifest.json", provenance)
            print("FINAL TEST", label, seed, evaluations[-1]["accuracy"], flush=True)


if __name__ == "__main__":
    main()
