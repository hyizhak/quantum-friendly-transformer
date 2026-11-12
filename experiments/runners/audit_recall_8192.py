"""Audit stored predictions and replay batches for the N8192 follow-up."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from experiments.lib.long_recall import input_certificate, matrix_records, paired_batch
from experiments.runners.reviewer_long_recall import make_model, state_hash
from experiments.runners.reviewer_recall_8192_recovery import initial_model


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


@torch.inference_mode()
def audit_run(run, device):
    config = json.loads((run / "config.json").read_text())
    provenance = json.loads((run / "manifest.json").read_text())
    result = json.loads((run / "result.json").read_text())
    assert provenance["config"] == config
    for path, expected in provenance["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    checkpoint_hash = hashlib.sha256((run / "final.pth").read_bytes()).hexdigest()
    assert checkpoint_hash == result["checkpoint_sha256"]
    saved = torch.load(run / "final.pth", map_location=device, weights_only=True)
    training_config = saved["config"]
    seed, variant = saved["seed"], saved["variant"]
    assert seed == result["seed"] and variant == result["variant"]
    assert training_config["curriculum"][-1]["N"] == 8192
    if result.get("reevaluation_only"):
        parent = ROOT / provenance["parent_training_run"]
        assert hashlib.sha256((parent / "final.pth").read_bytes()).hexdigest() == checkpoint_hash
        assert training_config == provenance["training_config"]
    else:
        assert training_config == config
        assert int(read_csv(run / "training.csv")[-1]["step"]) == config["steps"]
        for row in read_csv(run / "weights_training.csv"):
            assert float(row["effective_frobenius"]) <= config["cap"] * (1 + 1e-7)
    torch.manual_seed(seed)
    initial = initial_model(training_config, variant, device)
    assert state_hash(initial.state_dict()) == result["initial_state_sha256"]
    del initial
    model = make_model(training_config, variant, device)
    model.load_state_dict(saved["state_dict"])
    assert state_hash(model.state_dict()) == state_hash(saved["state_dict"])
    model.double().eval()
    weights = matrix_records(model)
    assert len(weights) == 6
    for field in ("training_input_row_squared_max", "training_input_matrix_squared_to_N_max"):
        assert 0 < result[field] <= 1
    predictions, metrics = read_csv(run / "predictions.csv"), read_csv(run / "test.csv")
    assert len(predictions) == config["test_pairs"] * 2 * len(config["test_lengths"])
    replayed, max_error = 0, 0.
    for evaluation in result["evaluation"]:
        n = evaluation["N"]
        rows = [r for r in predictions if int(r["N"]) == n]
        batches = [r for r in metrics if int(r["N"]) == n]
        assert len(rows) == evaluation["examples"] == 2 * config["test_pairs"]
        for i, row in enumerate(rows):
            assert int(row["example"]) == i and int(row["pair"]) == i // 2
            if i % 2:
                assert int(row["label"]) == 1 - int(rows[i-1]["label"])
                assert row["uniform_prediction"] == rows[i-1]["uniform_prediction"]
        for prefix in ["", "uniform_", "shuffled_query_"] + [f"noise_{e:g}_" for e in config["attention_noise_l2"]]:
            correct = sum(r[prefix+"prediction"] == r["label"] for r in rows)
            assert correct == sum(int(r[prefix+"correct"]) for r in batches)
            assert correct / len(rows) == evaluation[prefix+"accuracy"]
        assert evaluation["uniform_accuracy"] == .5
        if variant == "uniform":
            assert evaluation["accuracy"] == .5
        for batch in batches:
            assert float(batch["maximum_input_row_squared_norm"]) <= 1
            assert float(batch["maximum_input_matrix_squared_norm"]) <= n
            assert float(batch["maximum_paired_mean_gap"]) == 0
            assert float(batch["denominator"]) == (0 if variant == "uniform" else model.denominator(n))
            assert min(float(batch["minimum_sigma1"]), float(batch["minimum_sigma2"])) > 0
        generator = torch.Generator(device=device).manual_seed(config["test_seed"] + n)
        inputs, labels, selected = paired_batch(config["test_batch_pairs"], n, config["dimension"],
                                                config["key_dimension"], generator, device)
        input_certificate(inputs)
        inputs = inputs.double()
        assert torch.equal(inputs[::2].mean(1), inputs[1::2].mean(1))
        logits, diagnostics = model(inputs, diagnostics=True)
        index = torch.arange(len(labels), device=device)
        margin = logits[index, labels] - logits[index, 1-labels]
        for i, row in enumerate(rows[:len(labels)]):
            assert int(row["label"]) == int(labels[i])
            assert int(row["prediction"]) == int(logits[i].argmax())
            error = abs(float(row["margin"]) - float(margin[i]))
            assert error < 1e-8
            max_error = max(max_error, error)
        direct, hidden, _ = model.dense_forward(inputs[:4])
        assert float((direct-logits[:4]).abs().max()) < 1e-9
        assert float((hidden-diagnostics["hidden"][:4]).abs().max()) < 1e-9
        replayed += len(labels)
    return {"run": str(run.relative_to(ROOT)), "status": "passed", "seed": seed,
            "variant": variant, "cap": config["cap"], "evaluation_role": config["evaluation_role"],
            "stored_predictions_audited": len(predictions), "fresh_predictions_replayed": replayed,
            "maximum_replay_margin_error": max_error, "checkpoint_sha256": checkpoint_hash,
            "active_caps": sum(r["cap_active"] for r in weights),
            "maximum_effective_norm": max(r["effective_frobenius"] for r in weights)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directories", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reports = []
    for directory in args.directories:
        runs = sorted(directory.glob("*_seed*/result.json"))
        assert runs, directory
        for result in runs:
            report = audit_run(result.parent.resolve(), torch.device(args.device))
            reports.append(report)
            print("PASS", report["run"], flush=True)
    payload = {"status": "passed", "models": len(reports),
               "stored_predictions_audited": sum(r["stored_predictions_audited"] for r in reports),
               "fresh_predictions_replayed": sum(r["fresh_predictions_replayed"] for r in reports),
               "replay_scope": "First batch at each length for every model; not a second full evaluation.",
               "reports": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k:v for k,v in payload.items() if k != "reports"}, indent=2))


if __name__ == "__main__":
    main()
