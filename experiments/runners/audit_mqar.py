"""Audit all MQAR predictions and independently replay a batch per condition."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
from torch.nn import functional as F

from experiments.lib.mqar import MQARTransformer, state_hash, weight_records
from experiments.lib.experiment import save_json
from experiments.runners.mqar import load_data, sha256


def audit_run(run, assets, device):
    config = json.loads((run / "config.json").read_text())
    result = json.loads((run / "result.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    for filename, expected in manifest["source_sha256"].items():
        if sha256(filename) != expected:
            raise AssertionError(f"Changed source: {filename}")
        snapshot = run / "source_snapshot" / Path(filename).relative_to(ROOT)
        if sha256(snapshot) != expected:
            raise AssertionError(f"Changed snapshot: {snapshot}")
    if sha256(assets / "manifest.json") != manifest["data_manifest_sha256"]:
        raise AssertionError("Changed dataset manifest")
    if sha256(run / "final.pth") != result["checkpoint_sha256"]:
        raise AssertionError("Changed checkpoint")
    checkpoint = torch.load(run / "final.pth", weights_only=True, map_location=device)
    model = MQARTransformer(config, result["variant"]).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    if state_hash(model.state_dict()) != result["final_state_sha256"]:
        raise AssertionError("Restored state mismatch")
    norm_records = weight_records(model)
    reports = []
    with torch.inference_mode():
        for metrics in result["evaluation"]:
            name = f"test_N{metrics['N']}_K{metrics['pairs']}"
            data = load_data(assets, name)
            prediction_path = run / f"{name}_predictions.csv"
            with prediction_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            correct = sum(row["label"] == row["prediction"] for row in rows)
            uniform_correct = sum(row["label"] == row["uniform_prediction"] for row in rows)
            if len(rows) != metrics["queries"] or correct != metrics["correct"]:
                raise AssertionError(f"Accuracy mismatch: {run.name} {name}")
            if abs(uniform_correct / len(rows) - metrics["uniform_ablation_accuracy"]) > 1e-12:
                raise AssertionError("Ablation accuracy mismatch")
            coordinates = {(int(row["sequence"]), int(row["position"])) for row in rows}
            expected = set(map(tuple, (data["labels"] != -100).nonzero().tolist()))
            if coordinates != expected or len(coordinates) != len(rows):
                raise AssertionError("Scored query coverage mismatch")
            if any(int(data["labels"][int(row["sequence"]), int(row["position"])]) != int(row["label"]) for row in rows):
                raise AssertionError("Saved labels differ from official labels")
            count = min(config["eval_batch_size"], len(data["inputs"]))
            inputs, labels = data["inputs"][:count].to(device), data["labels"][:count].to(device)
            positions = (labels != -100).nonzero(as_tuple=True)
            logits = model(inputs, positions)
            predictions = logits.argmax(-1).cpu().tolist()
            ablations = model(inputs, positions, force_uniform=True).argmax(-1).cpu().tolist()
            stored = [row for row in rows if int(row["sequence"]) < count]
            if predictions != [int(row["prediction"]) for row in stored] or ablations != [int(row["uniform_prediction"]) for row in stored]:
                raise AssertionError("Fresh prediction replay mismatch")
            losses = F.cross_entropy(logits, labels[positions], reduction="none").cpu().tolist()
            error = max(abs(value - float(row["nll"])) for value, row in zip(losses, stored))
            if error > 1e-4:
                raise AssertionError(f"Replay NLL error: {error}")
            reports.append({"N":metrics["N"], "pairs":metrics["pairs"],
                            "stored_predictions_audited":len(rows), "fresh_queries_replayed":len(stored),
                            "maximum_replay_nll_error":error, "predictions_sha256":sha256(prediction_path)})
        # Independent dense attention check at trained weights in float64.
        model = model.double()
        inputs = data["inputs"][:1, :64].to(device)
        direct = model(inputs, dense=True)
        fast = model(inputs)
        dense_error = float((direct-fast).abs().max())
        if dense_error > 1e-8:
            raise AssertionError(f"Dense forward mismatch: {dense_error}")
    return {"run":run.name, "status":"passed", "dense_float64_max_error":dense_error,
            "capped_matrices":sum(r["capped"] for r in norm_records),
            "active_caps":sum(r["cap_active"] for r in norm_records), "conditions":reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = json.loads((args.assets / "config.json").read_text())
    reports = []
    for seed in config["seeds"]:
        starts = []
        for variant in config["variants"]:
            run = args.results / f"{variant}_seed{seed}"
            result = json.loads((run / "result.json").read_text())
            if result["smoke"]:
                raise AssertionError("Smoke result in benchmark cohort")
            starts.append(result["initial_state_sha256"])
            reports.append(audit_run(run, args.assets, torch.device(args.device)))
            print(f"PASS {run.name}", flush=True)
        if len(set(starts)) != 1:
            raise AssertionError("Unpaired initial weights")
    report = {"status":"passed", "models":len(reports), "paired_initializations":True,
              "all_stored_queries_audited":sum(c["stored_predictions_audited"] for r in reports for c in r["conditions"]),
              "fresh_queries_replayed":sum(c["fresh_queries_replayed"] for r in reports for c in r["conditions"]),
              "replay_scope":"First batch at every condition for every model; not a second full test evaluation",
              "reports":reports}
    save_json(args.results / "audit.json", report)
    print(json.dumps({k:v for k,v in report.items() if k != "reports"}, indent=2))


if __name__ == "__main__":
    main()
