"""Audit recovery artifacts and optionally replay saved checkpoints on a GPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import torch

from experiments.lib.recovery_extension import indices_hash, validate_scale_records
from experiments.lib.experiment import save_csv, save_json
from quantum_friendly_transformer.norm_transformer import collect_attention_scaling_stats


@torch.no_grad()
def measure_attention(model, data, device, limit=128, batch_size=32):
    """Inspect the final unmasked query on the first fixed evaluation examples."""
    model.eval()
    values = {}
    count = min(limit, len(data["labels"]))
    for start in range(0, count, batch_size):
        batch = {k: v[start:min(start + batch_size, count)] for k, v in data.items() if k != "example_id"}
        lengths = batch["attention_mask"].sum(-1)
        assert bool((lengths > 0).all())
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(**batch, output_attentions=True)
        for layer, weights in enumerate(output.attentions):
            raw = weights[torch.arange(len(lengths), device=device), :, lengths - 1, :].float()
            raw = raw * batch["attention_mask"][:, None, :]
            mass = raw.sum(-1)
            probability = raw / mass[..., None]
            visible_lengths = lengths
            if model.config.attention_layers[layer] == "local":
                visible_lengths = lengths.clamp(max=model.config.window_size)
            visible = visible_lengths[:, None].float()
            entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1)
            normalized_entropy = torch.where(visible > 1, entropy / visible.clamp_min(2).log(), torch.ones_like(entropy))
            statistics = torch.stack([probability.max(-1).values * visible, normalized_entropy,
                                      1 / probability.square().sum(-1) / visible, mass], dim=-1)
            values.setdefault(layer, []).append(statistics.reshape(-1, 4).cpu())
    rows = []
    for layer, blocks in values.items():
        measurements = torch.cat(blocks)
        row = {"layer": layer, "attention_type": model.config.attention_layers[layer],
               "evaluation_examples": count, "head_query_observations": len(measurements),
               "probabilities": "returned attention weights, renormalized after dtype rounding"}
        for i, name in enumerate(["visible_keys_times_max_probability", "normalized_entropy", "participation_fraction", "returned_weight_sum"]):
            for q, label in [(0., "minimum"), (.01, "p01"), (.5, "median"), (.95, "p95"), (1., "maximum")]:
                row[f"{name}_{label}"] = float(torch.quantile(measurements[:, i], q))
        rows.append(row)
    return rows


def validate(args):
    config = json.loads(args.config.read_text())
    requested_seeds = getattr(args, 'only_seeds', None)
    if requested_seeds is not None and not set(requested_seeds) <= set(config['seeds']):
        raise ValueError('Requested seed/candidate is absent from the recorded configuration')
    torch.set_num_threads(config["threads"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    reports, weights, missing, attention_rows = [], [], [], []
    tokenized_evaluation = {}
    for name in config["models"]:
        for mode in config["modes"]:
            for seed in config["seeds"]:
                if requested_seeds is not None and seed not in requested_seeds:
                    continue
                path = args.results / f"{name}_{mode}_seed{seed}"
                if not (path / "manifest.json").exists():
                    missing.append(path.name)
                    continue
                recorded = json.loads((path / "result.json").read_text())
                metadata = json.loads((path / "manifest.json").read_text())
                assert not recorded["smoke"]
                assert metadata["config"] == config
                references = list(config.get('parent_checkpoints', []))
                if 'parent_checkpoint' in config:
                    references.append(config['parent_checkpoint'])
                for reference in references:
                    assert hashlib.sha256(Path(reference['path']).read_bytes()).hexdigest() == reference['sha256']
                    parent = json.loads(Path(reference['result_path']).read_text())
                    assert parent['checkpoint_sha256'] == reference['sha256']
                    assert parent['name'] == name and parent['mode'] == mode
                    assert parent['configured_denominator_floor'] == recorded['configured_denominator_floor']
                for source, digest in metadata["source_sha256"].items():
                    assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest, source
                split = json.loads((path / "split.json").read_text())
                assert not set(split["indices"]["train"]) & set(split["indices"]["selection"])
                assert all(indices_hash(values) == split["sha256"][key] for key, values in split["indices"].items())
                prediction_rows = list(csv.DictReader((path / "predictions.csv").open()))
                assert len(prediction_rows) == 872
                correct = sum(r["label"] == r["prediction"] for r in prediction_rows)
                assert correct == recorded["evaluation"]["correct"]
                threshold = config["models"][name]["baseline_target"] - .01
                assert recorded["meets_historical_threshold"] == (correct / 872 >= threshold)
                selection = list(csv.DictReader((path / "selection.csv").open()))
                best = max(selection, key=lambda r: (float(r["accuracy"]), -float(r["mean_nll"])))
                assert int(best["step"]) == recorded["selected_step"]
                if 'selected_variant' in recorded:
                    assert best['variant'] == recorded['selected_variant']
                if recorded.get('external_teacher_used'):
                    reference = json.loads((path / 'teacher_reference.json').read_text())
                    teacher_path = Path(reference['manifest_path'])
                    assert hashlib.sha256(teacher_path.read_bytes()).hexdigest() == reference['manifest_sha256']
                    teacher = json.loads(teacher_path.read_text())
                    assert metadata['teacher'] == teacher
                    assert not reference['validation_targets_used_for_training']
                    cache_path = teacher_path.parent / 'training_logits.pth'
                    assert hashlib.sha256(cache_path.read_bytes()).hexdigest() == reference['training_logits_sha256']
                    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
                    assert cache['split'] == 'train' and len(cache['example_id']) == 67349
                    assert cache['example_id'].unique().numel() == 67349
                    assert teacher['dataset_revision'] == config['dataset_revision']
                    assert teacher['training_data_sha256'] == metadata['assets']['sha256']['data/sst2/train-00000-of-00001.parquet']
                    assert teacher['teacher_id'] == cache['teacher_id'] == reference['teacher_id']
                    assert teacher['teacher_revision'] == cache['teacher_revision'] == reference['teacher_revision']
                    assert reference['training_logits_sha256'] == teacher['training_logits_sha256']
                for table in path.glob("*.csv"):
                    for row in csv.DictReader(table.open()):
                        for value in row.values():
                            try:
                                number = float(value)
                            except (ValueError, TypeError):
                                continue
                            assert math.isfinite(number), (table, row)
                checkpoint_path = path / "selected.pth"
                assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == recorded["checkpoint_sha256"]
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                assert checkpoint["step"] == recorded["selected_step"]
                if 'selected_variant' in recorded:
                    assert checkpoint['selected_variant'] == recorded['selected_variant']
                assert checkpoint['config'] == config
                assert (checkpoint['model'], checkpoint['mode'], checkpoint['seed']) == (name, mode, seed)
                state = checkpoint["model_state_dict"]
                dense_scope_checked = False
                if metadata.get('train_scope') == 'all dense model parameters':
                    inventory = json.loads((path / 'parameter_inventory.json').read_text())
                    gradients = json.loads((path / 'first_step_gradient_audit.json').read_text())
                    names = {row['name'] for row in inventory['parameters']}
                    assert names == {row['name'] for row in gradients}
                    assert all(row['trainable'] for row in inventory['parameters'])
                    assert inventory['trainable_parameters'] == inventory['total_parameters']
                    assert inventory['total_parameters'] == sum(state[n].numel() for n in names)
                    assert not any('lora_' in n for n in state)
                    assert all(state[n].dtype == torch.float32 for n in names)
                    assert all(row['gradient_present'] and row['gradient_finite'] for row in gradients)
                    assert all(math.isfinite(row['gradient_norm']) for row in gradients)
                    dense_scope_checked = True
                frozen_parent_checked = False
                if metadata.get('train_scope') == 'existing classifier only':
                    parent_state = torch.load(config['parent_checkpoint']['path'], map_location='cpu',
                                              weights_only=True)['model_state_dict']
                    assert state.keys() == parent_state.keys()
                    assert all(torch.equal(value, parent_state[key]) for key, value in state.items()
                               if key != 'score.weight')
                    frozen_parent_checked = True
                    del parent_state
                cap_norms = []
                for key, raw in state.items():
                    if not key.endswith(".parametrizations.weight.original"):
                        continue
                    prefix = key.removesuffix("original")
                    gamma = config["cap"] * state[prefix + "0.g"].sigmoid()
                    raw_norm = torch.linalg.vector_norm(raw.float())
                    low_rank = prefix + '0.lora_A' in state
                    if low_rank:
                        recipe = config['models'][name]
                        combined = raw.float() + (recipe['low_rank_alpha'] / recipe['low_rank_rank']) * (
                            state[prefix + '0.lora_B'] @ state[prefix + '0.lora_A'])
                        composed_norm = torch.linalg.vector_norm(combined)
                        margin = 1 - 2 * torch.finfo(raw.dtype).eps
                        effective = (combined * torch.clamp(gamma * margin / (composed_norm + 1e-12), max=1.)).to(raw.dtype)
                    else:
                        composed_norm = raw_norm
                        effective = raw * torch.clamp(gamma / (raw_norm + 1e-12), max=1.)
                    effective_norm = float(torch.linalg.vector_norm(effective.float()))
                    assert effective_norm <= config["cap"] * (1 + 1e-6)
                    cap_norms.append(effective_norm)
                    weights.append({"run": path.name, "weight": key, "raw_frobenius_norm": float(raw_norm),
                                    "composed_frobenius_norm": float(composed_norm), "low_rank_update": low_rank,
                                    "gamma": float(gamma), "effective_frobenius_norm": effective_norm,
                                    "rounding_margin": margin if low_rank else 1.,
                                    "cap_active": bool(composed_norm > gamma * (margin if low_rank else 1.))})
                assert len(cap_norms) == recorded["applied_caps"]
                if mode == "bounded":
                    rows = list(csv.DictReader((path / "scales.csv").open()))
                    floor = recorded["configured_denominator_floor"]
                    assert all(float(r["score_scale"]) <= 1 / floor and float(r["score_denominator"]) >= floor for r in rows)
                    validate_scale_records(json.loads((path / "selected_scale_stats.json").read_text()),
                                           config["models"][name]["alpha_s"], config["cap"])
                report = {"run": path.name, "correct": correct, "examples": 872,
                          "selected_step": recorded["selected_step"], "source_hashes_match": True,
                          "checkpoint_hash_matches": True, "matrix_caps_checked": len(cap_norms),
                          "maximum_effective_weight_norm": max(cap_norms), "replayed": False}
                if frozen_parent_checked:
                    report['frozen_backbone_exactly_matches_parent'] = True
                if dense_scope_checked:
                    report.update(full_dense_parameter_scope_verified=True,
                                  trainable_parameters=inventory['trainable_parameters'],
                                  parameter_tensors_with_finite_gradients=len(gradients))
                if args.replay:
                    from transformers import AutoTokenizer
                    import pyarrow.parquet as pq
                    from experiments.runners.recovery_extension import build_model, evaluate
                    device = torch.device(args.device)
                    if name not in tokenized_evaluation:
                        tokenizer = AutoTokenizer.from_pretrained(args.assets / name, local_files_only=True)
                        tokenizer.pad_token = tokenizer.eos_token
                        tokenizer.padding_side = "right"
                        raw_rows = pq.read_table(args.assets / "data/sst2/validation-00000-of-00001.parquet").to_pylist()
                        batch = tokenizer([r["sentence"] for r in raw_rows], padding="max_length", truncation=True,
                                          max_length=config["sequence_length"], return_tensors="pt")
                        tokenized_evaluation[name] = {"input_ids": batch["input_ids"].to(device),
                                                     "attention_mask": batch["attention_mask"].to(device),
                                                     "labels": torch.tensor([r["label"] for r in raw_rows], device=device)}
                    model, _, _ = build_model(args.assets, name, config, seed, mode, device)
                    model.load_state_dict(state, strict=True)
                    scores, predictions, nlls, _ = evaluate(model, tokenized_evaluation[name], config["eval_batch_size"],
                                                           config["models"][name]["alpha_s"], device)
                    saved_predictions = torch.tensor([int(r["prediction"]) for r in prediction_rows])
                    saved_nlls = torch.tensor([float(r["nll"]) for r in prediction_rows])
                    assert torch.equal(predictions, saved_predictions), path
                    error = float((nlls - saved_nlls).abs().max())
                    assert error < 1e-4, (path, error)
                    assert scores["correct"] == correct
                    if mode == "bounded":
                        validate_scale_records(collect_attention_scaling_stats(model), config["models"][name]["alpha_s"], config["cap"])
                    report.update(replayed=True, all_predictions_match=True, maximum_per_example_nll_difference=error)
                    measured = measure_attention(model, tokenized_evaluation[name], device)
                    save_csv(path / "selected_attention_diagnostics.csv", measured)
                    attention_rows.extend({"run": path.name, **r} for r in measured)
                    del model
                    torch.cuda.empty_cache()
                reports.append(report)
                print(f"Validated {path.name}; replayed={report['replayed']}", flush=True)
    if missing and not args.allow_partial:
        raise ValueError("Missing runs: " + ", ".join(missing))
    suffix = getattr(args, 'output_suffix', '')
    report_name = ('checkpoint_replay' if args.replay else 'artifact_validation') + suffix + '.json'
    save_json(args.results / report_name,
              {"validated_runs": len(reports), "requested_seeds": requested_seeds,
               "missing": missing, "reports": reports})
    save_csv(args.results / ('selected_weight_norms' + suffix + '.csv'), weights)
    save_csv(args.results / ('selected_attention_diagnostics' + suffix + '.csv'), attention_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=(Path(__file__).resolve().parents[1] / 'configs/recovery/recovery_extension.json'))
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    validate(parser.parse_args())
