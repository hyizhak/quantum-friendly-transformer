"""Validation-only follow-up search with complete projection Frobenius bounds 1.

This is a new exploratory cohort, not a replacement for the completed original
seed-42 run. Execute using the original study's frozen source on PYTHONPATH.
Training never loads the test tensors. Added low-rank terms are INSIDE each
complete matrix's normalization. No attention-temperature changes are made.
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import get_cosine_schedule_with_warmup

from experiments.rigorous_rerun import run as base
from quantum_friendly_transformer.trainer.checkpoints import load_unparametrized_checkpoint


JOB = {"id": "genomic/42/multilayer/fn", "dataset": "genomic", "seed": 42,
       "family": "multilayer", "variant": "fn"}
MARGIN = 1e-6


class UnitAdapter(nn.Module):
    """Normalize W + BA as ONE matrix; optional scalar is also bounded by 1."""

    def __init__(self, shape, rank=0, learn_gamma=False):
        super().__init__()
        self.rank = rank
        if rank:
            self.a = nn.Parameter(torch.randn(rank, shape[1]) * 0.02)
            self.b = nn.Parameter(torch.zeros(shape[0], rank))
        if learn_gamma:
            self.g = nn.Parameter(torch.tensor(math.log(0.99 / 0.01)))

    def forward(self, weight):
        # Keep the complete effective matrix and its reduction in FP32 under AMP.
        with torch.autocast(device_type=weight.device.type, enabled=False):
            value = weight.float()
            if self.rank:
                value = value + self.b.float() @ self.a.float()
            gain = torch.sigmoid(self.g.float()) if hasattr(self, "g") else 1.0
            return ((1 - MARGIN) * gain * value /
                    (torch.linalg.vector_norm(value) + 1e-12)).to(weight.dtype)


class Classifier(nn.Module):
    def __init__(self, network, pool):
        super().__init__()
        self.network = network
        self.pool = pool
        if pool == "convex":
            self.pool_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids, attention_mask):
        sequence, pooled = self.network.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = sequence[:, 0]
        feature = cls
        if self.pool == "convex":
            mask = attention_mask.unsqueeze(-1).to(sequence.dtype)
            mean = (sequence * mask).sum(1) / mask.sum(1).clamp_min(1)
            gate = torch.sigmoid(self.pool_logit)
            # A convex combination cannot exceed the largest input row norm.
            feature = gate * cls + (1 - gate) * mean
            pooled = self.network.bert.pooler(feature, pool=False)
        logits = self.network.classifier(self.network.dropout(pooled))
        return SimpleNamespace(logits=logits, feature=feature)


def load_plan(root):
    plan = json.loads((root / "plan.json").read_text())
    if base.file_sha(__file__) != plan["runner_sha256"]:
        raise RuntimeError("Search runner differs from the frozen plan")
    original = Path(plan["original_study"])
    study = json.loads((original / "study.json").read_text())
    assert study["study_sha256"] == plan["original_study_sha256"]
    for split in ("train", "validation"):
        assert base.file_sha(original / "data/genomic" / f"{split}.pt") == plan["data_sha256"][split]
    for name in ("student", "teacher"):
        assert base.file_sha(Path(plan[f"{name}_checkpoint"])) == plan[f"{name}_sha256"]
    return plan, original, study


def build(plan, original, study, trial):
    network, _ = base.build_model(study, original, JOB)
    if trial["initialization"] == "fn25":
        state = torch.load(plan["student_checkpoint"], map_location="cpu", weights_only=True)
        network.load_state_dict(state["model"], strict=True)
    elif trial["initialization"] == "teacher12":
        state = torch.load(plan["teacher_checkpoint"], map_location="cpu", weights_only=True)
        load_unparametrized_checkpoint(network, state["model"])
    elif trial["initialization"] != "pretrained":
        raise ValueError(trial["initialization"])
    base.seed_everything(42)
    count = 0
    for module in network.modules():
        if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
            raw = module.parametrizations.weight.original
            module.parametrizations.weight[0] = UnitAdapter(raw.shape, trial["rank"], trial["learn_gamma"])
            if trial["rank"]:
                raw.requires_grad_(False)
            count += 1
    assert count == 72
    if trial["freeze_embeddings"]:
        network.bert.embeddings.word_embeddings.weight.requires_grad_(False)
    for module in network.modules():
        if isinstance(module, nn.Dropout):
            module.p = trial["dropout"]
        # Attention's branch selector must retain the existing nonzero dropout.
        if hasattr(module, "p_dropout"):
            module.p_dropout = trial["dropout"]
    model = Classifier(network, trial["pool"])
    return model


@torch.inference_mode()
def audit(model):
    model.eval()
    rows = []
    for name, module in model.named_modules():
        if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
            transform = module.parametrizations.weight[0]
            norm = torch.linalg.vector_norm(module.weight.double()).item()
            if not math.isfinite(norm) or norm > 1.0:
                raise AssertionError(f"Complete effective matrix exceeds 1: {name}={norm}")
            rows.append({"module": name, "shape": list(module.weight.shape), "frobenius_fp64": norm,
                         "rank": transform.rank,
                         "gamma": torch.sigmoid(transform.g).item() if hasattr(transform, "g") else 1.0})
    assert len(rows) == 72
    return rows


def prepare(root, original, repository):
    root.mkdir(parents=True, exist_ok=False)
    base.configure()
    study = json.loads((original / "study.json").read_text())
    base.verify_study(study, original)
    common = {"seed": 42, "batch_size": 128, "eval_batch_size": 256, "weight_decay": 0.01,
              "gradient_clip": 1.0, "warmup_ratio": 0.05, "temperature": 2.0,
              "smoothing": 0.02, "rank": 0, "learn_gamma": False, "freeze_embeddings": False,
              "dropout": 0.1, "feature_weight": 0.0, "pool": "cls"}
    definitions = [
        {"name": "fn1_recovery_kd", "worker": 0, "initialization": "fn25", "lr": 2e-5,
         "aux_lr": 1e-4, "epochs": 30, "min_epochs": 10, "patience": 8, "kd_weight": 0.5},
        {"name": "fn1_teacher_pool_kd", "worker": 1, "initialization": "teacher12", "lr": 5e-5,
         "aux_lr": 2e-4, "epochs": 40, "min_epochs": 15, "patience": 10, "kd_weight": 0.5,
         "pool": "convex", "feature_weight": 0.1},
        {"name": "fn1_lora_pool_kd", "worker": 0, "initialization": "fn25", "lr": 1e-4,
         "aux_lr": 5e-4, "epochs": 35, "min_epochs": 15, "patience": 10, "kd_weight": 0.5,
         "pool": "convex", "rank": 8, "learn_gamma": True, "freeze_embeddings": True},
        {"name": "fn1_lower_lr", "worker": 1, "initialization": "pretrained", "lr": 5e-5,
         "aux_lr": 2e-4, "epochs": 40, "min_epochs": 20, "patience": 10, "kd_weight": 0.0,
         "dropout": 0.2, "smoothing": 0.05},
    ]
    trials = [{**common, **d} for d in definitions]
    student = original / "runs/genomic/42/multilayer/fn/best.pt"
    teacher = original / "runs/genomic/42/multilayer/vanilla/best.pt"
    teacher_history = json.loads((teacher.parent / "history.json").read_text())
    teacher_choice = max(teacher_history[1:], key=lambda x: x["validation"]["accuracy"])
    assert teacher_choice["epoch"] == 12
    snapshot = root / "unit_fn_search.py"
    shutil.copy2(__file__, snapshot)
    plan = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "original_study": str(original), "original_study_sha256": study["study_sha256"],
            "repository": str(repository), "runner_sha256": base.file_sha(snapshot),
            "student_checkpoint": str(student), "student_sha256": base.file_sha(student),
            "teacher_checkpoint": str(teacher), "teacher_sha256": base.file_sha(teacher),
            "teacher_selection": teacher_choice,
            "data_sha256": {s: study["data"]["genomic"]["splits"][s]["tensor_sha256"] for s in ("train", "validation", "test")},
            "constraints": {"projection_frobenius_max": 1.0, "effective_matrix_margin": MARGIN,
                            "projection_count": 72, "attention_score_multiplier": 8.0,
                            "layernorm": "Original fixed unit-affine LayerNorm retained",
                            "normalization_scope": "Complete Q/K/V/output/gated-FFN/down matrices; original embedding/pooler/classifier exclusions retained. No additional unnormalized readout matrices.",
                            "adapter": "W+BA is jointly normalized, not a sum of separately normalized matrices.",
                            "pooling": "Convex CLS/mean pooling where stated; no gain exceeding one."},
            "selection": "Best validation accuracy, first checkpoint wins ties, includes epoch zero. No trial test evaluations. Final test only after locking the chosen variant.",
            "prior_test_disclosure": "The original seed42 test scores were already observed. This is a disclosed follow-up optimization, not a fresh untouched-test confirmatory study.",
            "trials": trials}
    base.write_json(root / "plan.json", plan)
    # The teacher is used only to form targets for TRAINING examples.
    base.seed_everything(42)
    model = base.baseline_multilayer(study)
    state = torch.load(teacher, map_location="cpu", weights_only=True)
    assert state["epoch"] == 12 and state["validation"] == teacher_choice["validation"]
    model.load_state_dict(state["model"], strict=True)
    model.cuda().eval()
    data = torch.load(original / "data/genomic/train.pt", weights_only=True)
    logits, features = [], []
    with torch.inference_mode():
        for _, batch in base.batches(data, 256):
            sequence, pooled = model.bert(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            scores = model.classifier(model.dropout(pooled)).float()
            logits.append(scores.cpu())
            features.append(F.normalize(sequence[:, 0].float(), dim=-1).cpu())
    teacher_data = {"logits": torch.cat(logits), "feature": torch.cat(features)}
    assert teacher_data["logits"].shape == (24387, 2)
    assert torch.isfinite(teacher_data["logits"]).all() and torch.isfinite(teacher_data["feature"]).all()
    base.save_checkpoint(root / "teacher_train.pt", teacher_data)
    base.write_json(root / "teacher_cache.json", {"split": "train", "rows": 24387,
                    "sha256": base.file_sha(root / "teacher_train.pt"),
                    "teacher_checkpoint_sha256": plan["teacher_sha256"],
                    "training_accuracy": float((teacher_data["logits"].argmax(-1) == data["labels"]).float().mean()),
                    "test_examples_accessed": 0})
    shutil.copy2(root / "plan.json", repository / "artifacts/rigorous_rerun/unit_fn_search_plan.json")
    print(json.dumps({"prepared": str(root), "trials": [t["name"] for t in trials], "max_frobenius": 1}), flush=True)


def optimizer(model, trial):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        auxiliary = any(s in name for s in ("pooler.", "classifier.", "pool_logit", "residual_scale", ".0.a", ".0.b", ".0.g"))
        lr = trial["aux_lr"] if auxiliary else trial["lr"]
        decay = trial["weight_decay"] if parameter.ndim >= 2 and not auxiliary else 0.0
        groups.setdefault((lr, decay), []).append(parameter)
    return torch.optim.AdamW([{"params": p, "lr": lr, "weight_decay": wd} for (lr, wd), p in groups.items()],
                             betas=(0.9, 0.999), eps=1e-8, foreach=False)


def train(root, plan, original, study, trial):
    folder = root / trial["name"]
    folder.mkdir(exist_ok=True)
    if (folder / "selected.json").exists():
        return
    model = build(plan, original, study, trial).cuda()
    train_data = torch.load(original / "data/genomic/train.pt", weights_only=True)
    val_data = torch.load(original / "data/genomic/validation.pt", weights_only=True)
    cache_record = json.loads((root / "teacher_cache.json").read_text())
    assert base.file_sha(root / "teacher_train.pt") == cache_record["sha256"]
    teacher = torch.load(root / "teacher_train.pt", weights_only=True)
    opt = optimizer(model, trial)
    steps = math.ceil(len(train_data["labels"]) / trial["batch_size"]) * trial["epochs"]
    scheduler = get_cosine_schedule_with_warmup(opt, math.ceil(steps * trial["warmup_ratio"]), steps)
    latest = folder / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        assert state["trial"] == trial and state["plan_sha256"] == base.file_sha(root / "plan.json")
        model.load_state_dict(state["model"], strict=True);opt.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        history, best, stale = state["history"], state["best"], state["stale"]
        base.restore_rng(state["rng"])
        del state
    else:
        norms = audit(model)
        metrics, _, _ = base.evaluate(model, val_data, JOB, trial["eval_batch_size"])
        # Full-model padding/batch invariance check uses validation examples only.
        few = {k: v[:4] for k, v in val_data.items()}
        _, _, a = base.evaluate(model, few, JOB, 4, return_logits=True)
        _, _, b = base.evaluate(model, few, JOB, 1, return_logits=True)
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)
        history, best, stale = [{"epoch": 0, "validation": metrics}], metrics["accuracy"], 0
        base.save_checkpoint(folder / "best.pt", {"model": model.state_dict(), "epoch": 0, "validation": metrics, "trial": trial})
        base.write_json(folder / "initial_audit.json", {"norms": norms, "batch_invariance_passed": True,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                        "test_examples_accessed": 0})
        print(json.dumps({"trial": trial["name"], **history[0]}), flush=True)
        base.seed_everything(42)
    base.write_json(folder / "run.json", {"trial": trial, "pid": os.getpid(), "gpu": torch.cuda.get_device_name(),
                    "plan_sha256": base.file_sha(root / "plan.json")})
    for epoch in range(history[-1]["epoch"] + 1, trial["epochs"] + 1):
        if history[-1]["epoch"] >= trial["min_epochs"] and stale >= trial["patience"]:
            break
        model.train();started = time.monotonic();total = 0.0;max_grad = 0.0
        order = base.epoch_order(len(train_data["labels"]), 42, 1000 + epoch)
        for indices, batch in base.batches(train_data, trial["batch_size"], order):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = result.logits.float()
                ce = F.cross_entropy(logits, batch["labels"], label_smoothing=trial["smoothing"])
                loss = ce
                if trial["kd_weight"]:
                    temperature = trial["temperature"]
                    target = teacher["logits"][indices].cuda()
                    kd = F.kl_div(F.log_softmax(logits / temperature, dim=-1),
                                  F.softmax(target / temperature, dim=-1), reduction="batchmean") * temperature**2
                    loss = (1 - trial["kd_weight"]) * ce + trial["kd_weight"] * kd
                if trial["feature_weight"]:
                    target_feature = teacher["feature"][indices].cuda()
                    cosine = (F.normalize(result.feature.float(), dim=-1) * target_feature).sum(-1)
                    loss = loss + trial["feature_weight"] * (1 - cosine).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite follow-up training loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
            max_grad = max(max_grad, grad.item());opt.step();scheduler.step()
            total += loss.item() * len(indices)
        metrics, _, _ = base.evaluate(model, val_data, JOB, trial["eval_batch_size"])
        norms = audit(model)
        if metrics["accuracy"] > best:
            best, stale = metrics["accuracy"], 0
            base.save_checkpoint(folder / "best.pt", {"model": model.state_dict(), "epoch": epoch,
                                 "validation": metrics, "trial": trial})
        else:
            stale += 1
        record = {"epoch": epoch, "validation": metrics, "train_objective": total / len(train_data["labels"]),
                  "max_gradient_norm": max_grad, "maximum_effective_frobenius_fp64": max(r["frobenius_fp64"] for r in norms),
                  "seconds": time.monotonic() - started, "best_validation_accuracy": best}
        if model.pool == "convex":
            record["cls_pool_weight"] = torch.sigmoid(model.pool_logit).item()
        history.append(record)
        base.save_checkpoint(latest, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                             "scheduler": scheduler.state_dict(), "rng": base.capture_rng(),
                             "history": history, "best": best, "stale": stale, "trial": trial,
                             "plan_sha256": base.file_sha(root / "plan.json")})
        base.write_json(folder / "history.json", history)
        print(json.dumps({"trial": trial["name"], **record}), flush=True)
    selected = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model"], strict=True)
    metrics, predictions, logits = base.evaluate(model, val_data, JOB, trial["eval_batch_size"], return_logits=True)
    assert metrics == selected["validation"]
    np.savez_compressed(folder / "validation_predictions.npz", labels=val_data["labels"].numpy(),
                        predictions=predictions.numpy(), logits=logits.numpy())
    norms = audit(model)
    base.write_json(folder / "selected.json", {"trial": trial, "trained_epochs": history[-1]["epoch"],
                    "selected_epoch": selected["epoch"], "validation": metrics, "norms": norms,
                    "checkpoint_sha256": base.file_sha(folder / "best.pt"),
                    "validation_predictions_sha256": base.file_sha(folder / "validation_predictions.npz"),
                    "test_evaluated": False})
    print(json.dumps({"selected": trial["name"], "epoch": selected["epoch"], "validation": metrics}), flush=True)
    del model, opt, scheduler, selected, teacher, train_data, val_data, logits, predictions
    gc.collect();torch.cuda.empty_cache()


def worker(root, worker_id):
    base.configure()
    plan, original, study = load_plan(root)
    for trial in plan["trials"]:
        if trial["worker"] == worker_id:
            train(root, plan, original, study, trial)
    base.write_json(root / f"worker_{worker_id}_complete.json", {"worker": worker_id, "test_evaluated": False})


def status(root):
    plan = json.loads((root / "plan.json").read_text())
    rows = []
    for trial in plan["trials"]:
        folder = root / trial["name"]
        if (folder / "selected.json").exists():
            d = json.loads((folder / "selected.json").read_text())
            rows.append({"trial": trial["name"], "phase": "validation_complete", "epochs": d["trained_epochs"],
                         "selected_epoch": d["selected_epoch"], "validation_accuracy": d["validation"]["accuracy"]})
        elif (folder / "history.json").exists():
            h = json.loads((folder / "history.json").read_text())
            rows.append({"trial": trial["name"], "phase": "training", "epoch": h[-1]["epoch"],
                         "best_validation_accuracy": max(x["validation"]["accuracy"] for x in h)})
        else:
            rows.append({"trial": trial["name"], "phase": "not_started"})
    print(json.dumps(rows, indent=2), flush=True)


def finalize(root):
    """Evaluate exactly one variant, after all validation selections are fixed."""
    base.configure()
    plan, original, study = load_plan(root)
    candidates = []
    for trial in plan["trials"]:
        folder = root / trial["name"]
        record = json.loads((folder / "selected.json").read_text())
        assert base.file_sha(folder / "best.pt") == record["checkpoint_sha256"]
        candidates.append(record)
    chosen = max(candidates, key=lambda x: x["validation"]["accuracy"])
    lock = {"plan_sha256": base.file_sha(root / "plan.json"), "candidates": candidates,
            "chosen": chosen["trial"]["name"], "rule": plan["selection"],
            "prior_test_disclosure": plan["prior_test_disclosure"]}
    lock_path = root / "selection_lock.json"
    if lock_path.exists():
        assert json.loads(lock_path.read_text()) == lock
    else:
        base.write_json(lock_path, lock)
    trial = chosen["trial"]
    output = root / "final"
    output.mkdir(exist_ok=True)
    report_path = output / "result.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        assert report["selection_lock_sha256"] == base.file_sha(lock_path)
        assert report["predictions_sha256"] == base.file_sha(output / "test_predictions.npz")
    else:
        model = build(plan, original, study, trial).cuda()
        state = torch.load(root / trial["name"] / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        validation = torch.load(original / "data/genomic/validation.pt", weights_only=True)
        metrics, _, _ = base.evaluate(model, validation, JOB, trial["eval_batch_size"])
        # A different A100 variant may change last-bit loss reductions. All
        # validation predictions must reproduce the selection before test use.
        old = np.load(root / trial["name"] / "validation_predictions.npz")
        _, predictions, _ = base.evaluate(model, validation, JOB, trial["eval_batch_size"])
        assert np.array_equal(predictions.numpy(), old["predictions"])
        assert metrics["accuracy"] == chosen["validation"]["accuracy"]
        test_path = original / "data/genomic/test.pt"
        assert base.file_sha(test_path) == plan["data_sha256"]["test"]
        test = torch.load(test_path, weights_only=True)
        metrics, predictions, logits = base.evaluate(model, test, JOB, trial["eval_batch_size"], return_logits=True)
        y, p = test["labels"].numpy(), predictions.numpy()
        assert len(y) == 9034 and logits.shape == (9034, 2) and torch.isfinite(logits).all()
        counts = {"tp": int(((y == 1) & (p == 1)).sum()), "tn": int(((y == 0) & (p == 0)).sum()),
                  "fp": int(((y == 0) & (p == 1)).sum()), "fn": int(((y == 1) & (p == 0)).sum())}
        assert metrics["accuracy"] == (counts["tp"] + counts["tn"]) / 9034
        assert abs(metrics["f1"] - 2 * counts["tp"] / (2 * counts["tp"] + counts["fp"] + counts["fn"])) < 1e-12
        norms = audit(model)
        np.savez_compressed(output / "test_predictions.npz", labels=y, predictions=p, logits=logits.numpy())
        report = {"trial": trial, "selected_epoch": chosen["selected_epoch"],
                  "trained_epochs": chosen["trained_epochs"], "validation": chosen["validation"],
                  "test": metrics, "confusion_counts": counts, "norms": norms,
                  "checkpoint_sha256": chosen["checkpoint_sha256"],
                  "selection_lock_sha256": base.file_sha(lock_path),
                  "predictions_sha256": base.file_sha(output / "test_predictions.npz"),
                  "prior_test_disclosure": plan["prior_test_disclosure"],
                  "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        base.write_json(report_path, report)
    repository = Path(plan["repository"])
    exported = repository / "artifacts/rigorous_rerun/unit_fn_search"
    shutil.copytree(output, exported, dirs_exist_ok=True)
    shutil.copy2(root / "plan.json", exported / "plan.json")
    shutil.copy2(lock_path, exported / "selection_lock.json")
    for candidate in candidates:
        name = candidate["trial"]["name"]
        shutil.copy2(root / name / "selected.json", exported / f"{name}_validation.json")
        shutil.copy2(root / name / "history.json", exported / f"{name}_history.json")
    base.write_json(root / "complete.json", {"test": report["test"],
                    "report": str(exported / "result.json"), "frobenius_max": 1.0})
    print(json.dumps({"complete": True, "selected": trial["name"], "test": report["test"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "worker", "status", "finalize"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root.resolve(), args.original.resolve(), args.repository.resolve())
    elif args.action == "worker":
        worker(args.root.resolve(), args.worker)
    elif args.action == "finalize":
        finalize(args.root.resolve())
    else:
        status(args.root.resolve())
