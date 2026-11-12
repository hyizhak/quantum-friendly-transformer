"""Audit all long-recall records and freshly replay one batch per model/length.

The complete stored predictions are audited. Fresh inference is explicitly a
partial replay (the first batch at each length), not a full test-set replay.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
from torch.nn import functional as F

from experiments.lib.long_recall import input_certificate, matrix_records, paired_batch
from experiments.lib.reviewer import save_json
from experiments.runners.reviewer_long_recall import make_model, state_hash

COHORTS = ("long_recall_1024", "long_recall_2048", "long_recall_routing_1024", "long_recall_routing_2048")


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b, tolerance=1e-10):
    assert math.isfinite(float(a)) and math.isfinite(float(b)), (a, b)
    assert abs(float(a) - float(b)) <= tolerance, (a, b, tolerance)


@torch.inference_mode()
def validate(root, device):
    reports, initial_hashes, test_hashes, sources = [], {}, {}, {}
    for cohort in COHORTS:
        directory = root / cohort
        plan = json.loads((directory / "plan.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        assert config["seeds"] == [11, 22, 33]
        assert len(plan) == len(config["seeds"]) * len(config["variants"]) == 12
        assert {(x["seed"], x["variant"]) for x in plan} == {
            (s, v) for s in config["seeds"] for v in config["variants"]}
        assert config["steps"] == 3000 and config["curriculum"][-1]["through_step"] == 3000
        assert config["test_pairs"] == 2048 and len(config["test_lengths"]) == 6
        for item in plan:
            assert item["status"] == "completed", (cohort, item)
            run = directory / f"{item['variant']}_seed{item['seed']}"
            result = json.loads((run / "result.json").read_text())
            manifest = json.loads((run / "manifest.json").read_text())
            assert json.loads((run / "config.json").read_text()) == config == manifest["config"]
            assert not result["smoke"] and not manifest["smoke"]
            assert result["seed"] == manifest["seed"] == item["seed"]
            assert result["variant"] == manifest["variant"] == item["variant"]
            for recorded_path, checksum in manifest["source_sha256"].items():
                path = Path(recorded_path)
                if not path.exists():
                    # Permit relocation of an otherwise byte-identical artifact.
                    parts = path.parts
                    index = next(i for i, x in enumerate(parts) if x in ("experiments", "src"))
                    path = ROOT.joinpath(*parts[index:])
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                assert actual == checksum, str(path)
                sources[str(path.relative_to(ROOT))] = actual
            assert hashlib.sha256((run / "final.pth").read_bytes()).hexdigest() == result["checkpoint_sha256"]
            checkpoint = torch.load(run / "final.pth", map_location=device, weights_only=True)
            assert checkpoint["config"] == config
            assert checkpoint["seed"] == item["seed"] and checkpoint["variant"] == item["variant"]
            torch.manual_seed(item["seed"])
            model = make_model(config, item["variant"], device)
            actual_initial_hash = state_hash(model.state_dict())
            assert actual_initial_hash == result["initial_state_sha256"]
            if item["seed"] in initial_hashes:
                assert initial_hashes[item["seed"]] == actual_initial_hash
            initial_hashes[item["seed"]] = actual_initial_hash
            model.load_state_dict(checkpoint["state_dict"])
            assert state_hash(model.state_dict()) == state_hash(checkpoint["state_dict"])
            model.double().eval()
            for layer in (model.ln1, model.ln2):
                assert layer.eps == 0 and layer.affine_scale == 1 / math.sqrt(config["dimension"])
                assert not list(layer.parameters()) and torch.count_nonzero(layer.bias) == 0
            weights = csv_rows(run / "weights_evaluation_float64.csv")
            actual_weights = matrix_records(model)
            assert len(weights) == len(actual_weights) == 6
            for recorded, actual in zip(weights, actual_weights):
                assert recorded["matrix"] == actual["matrix"]
                close(recorded["effective_frobenius"], actual["effective_frobenius"], 1e-6)
                assert actual["effective_frobenius"] <= config["cap"] * (1 + 1e-7)
            for row in csv_rows(run / "weights_training.csv"):
                assert float(row["effective_frobenius"]) <= config["cap"] * (1 + 1e-7)
            training = csv_rows(run / "training.csv")
            assert int(training[-1]["step"]) == result["training_steps"] == config["steps"]
            for key in ("training_input_row_squared_max", "training_input_matrix_squared_to_N_max"):
                assert 0 < result[key] <= 1
            for key in ("training_sigma1_min", "training_sigma2_min"):
                assert result[key] > 0
            metrics, predictions = csv_rows(run / "test.csv"), csv_rows(run / "predictions.csv")
            assert len(predictions) == len(config["test_lengths"]) * 2 * config["test_pairs"]
            assert len(metrics) == len(config["test_lengths"]) * config["test_pairs"] // config["test_batch_pairs"]
            max_replay_error, max_dense_error, replay_count = 0., 0., 0
            for evaluation in result["evaluation"]:
                n = evaluation["N"]
                examples = [x for x in predictions if int(x["N"]) == n]
                batches = [x for x in metrics if int(x["N"]) == n]
                count = len(examples)
                assert count == evaluation["examples"] == 2 * config["test_pairs"]
                assert [int(x["example"]) for x in examples] == list(range(count))
                signature = hashlib.sha256()
                for i, row in enumerate(examples):
                    assert int(row["pair"]) == i // 2
                    signature.update(bytes([int(row["label"])]))
                    for key in ("nll", "margin", "target_probability"):
                        assert math.isfinite(float(row[key]))
                    assert 0 <= float(row["target_probability"]) <= 1
                    if i % 2:
                        assert int(row["label"]) == 1 - int(examples[i-1]["label"])
                        assert row["uniform_prediction"] == examples[i-1]["uniform_prediction"]
                identity = (config["test_seed"], n)
                if identity in test_hashes:
                    assert test_hashes[identity] == signature.hexdigest()
                test_hashes[identity] = signature.hexdigest()
                for prefix in ("", "uniform_", "shuffled_query_") + tuple(f"noise_{e:g}_" for e in config["attention_noise_l2"]):
                    correct = sum(x[prefix + "prediction"] == x["label"] for x in examples)
                    assert correct == sum(int(x[prefix + "correct"]) for x in batches)
                    close(correct / count, evaluation[prefix + "accuracy"])
                close(evaluation["uniform_accuracy"], .5, 0)
                if item["variant"] == "uniform":
                    close(evaluation["accuracy"], .5, 0)
                for batch in batches:
                    assert float(batch["maximum_input_row_squared_norm"]) <= 1
                    assert float(batch["maximum_input_matrix_squared_norm"]) <= n
                    assert float(batch["maximum_paired_mean_gap"]) == 0
                    assert min(float(batch["minimum_sigma1"]), float(batch["minimum_sigma2"])) > 0
                    if item["variant"] == "strict_frobenius":
                        assert float(batch["maximum_absolute_logit"]) <= 1 / n + 1e-12
                        close(batch["kappa"], 1, 0)
                # Regenerate the FIRST test batch, including its training-independent RNG.
                generator = torch.Generator(device=device).manual_seed(config["test_seed"] + n)
                rows, labels, selected = paired_batch(config["test_batch_pairs"], n, config["dimension"], config["key_dimension"], generator, device)
                input_certificate(rows)
                rows = rows.double()
                assert torch.equal(rows[::2].mean(1), rows[1::2].mean(1))
                logits, diag = model(rows, diagnostics=True)
                uniform = model(rows, force_uniform=True)
                assert torch.equal(uniform[::2], uniform[1::2])
                batch_index = torch.arange(len(labels), device=device)
                margin = logits[batch_index, labels] - logits[batch_index, 1-labels]
                nll = F.cross_entropy(logits, labels, reduction="none")
                target_probability = diag["probabilities"][batch_index, selected]
                pred = logits.argmax(-1)
                for i, recorded in enumerate(examples[:len(labels)]):
                    assert int(recorded["label"]) == int(labels[i])
                    assert int(recorded["prediction"]) == int(pred[i])
                    for key, values in (("margin", margin), ("nll", nll), ("target_probability", target_probability)):
                        error = abs(float(recorded[key]) - float(values[i]))
                        max_replay_error = max(max_replay_error, error)
                        assert error < 1e-8, (run, n, i, key, error)
                dense, dense_hidden, _ = model.dense_forward(rows[:4])
                dense_error = max(float((dense - logits[:4]).abs().max()),
                                  float((dense_hidden - diag["hidden"][:4]).abs().max()))
                max_dense_error = max(max_dense_error, dense_error)
                assert dense_error < 1e-9
                replay_count += len(labels)
            report = {"cohort": cohort, "seed": item["seed"], "variant": item["variant"],
                      "stored_examples_audited": len(predictions), "fresh_examples_replayed": replay_count,
                      "maximum_replay_error": max_replay_error, "maximum_dense_error": max_dense_error,
                      "checkpoint_sha256": result["checkpoint_sha256"]}
            reports.append(report)
            print(f"PASS {cohort}/{run.name}: {len(predictions)} stored; {replay_count} fresh", flush=True)
    output = root / "long_recall_summary"
    output.mkdir(exist_ok=True)
    save_json(output / "audit.json", {"status": "passed", "models": len(reports),
        "stored_examples_audited": sum(x["stored_examples_audited"] for x in reports),
        "fresh_examples_replayed": sum(x["fresh_examples_replayed"] for x in reports),
        "fresh_replay_scope": "First batch at all six lengths for every model; not a full independent evaluation",
        "source_sha256": sources, "reports": reports})
    print(f"Validated all {len(reports)} models.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/reviewer")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    validate(args.results.resolve(), torch.device(args.device))
