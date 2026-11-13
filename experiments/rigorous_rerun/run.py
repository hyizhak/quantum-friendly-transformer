"""Deterministic, paired and resumable reruns; test evaluation is a separate phase.

Run with HF_HOME on the data volume and CUBLAS_WORKSPACE_CONFIG=:4096:8.
The queue reads a frozen study manifest and never chooses settings using test scores.
"""

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    BertConfig,
    get_cosine_schedule_with_warmup,
)

from quantum_friendly_transformer.norm_transformer import (
    frobenius_normalized_transformer as fn,
)
from quantum_friendly_transformer.norm_transformer import (
    spectral_normalized_transformer as sn,
)
from quantum_friendly_transformer.norm_transformer.bert.frobenius_normalized_bert import (
    FrobeniuslyNormalizedBertConfig,
    FrobeniuslyNormalizedBertForSequenceClassification,
)
from quantum_friendly_transformer.norm_transformer.normalization import (
    UnitNormLayerNorm,
)
from quantum_friendly_transformer.trainer.checkpoints import (
    load_unparametrized_checkpoint,
)

from .prepare import digest, file_sha


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Distinct temporary files also make concurrent test-lock creation safe.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(json.dumps(value, indent=2, allow_nan=False))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def epoch_order(size, seed, epoch):
    # Separate generator: dropout consumption cannot change the sample order.
    generator = torch.Generator().manual_seed(seed * 1000003 + epoch)
    return torch.randperm(size, generator=generator)


def settings(protocol, dataset, family):
    value = copy.deepcopy(protocol[family])
    if family == "single":
        value.update(value[dataset])
        for key in protocol["datasets"]:
            value.pop(key, None)
    return value


def jobs(protocol):
    result = []
    group = 0
    for dataset in protocol["datasets"]:
        for seed in protocol["seeds"]:
            for family in ("single", "fine_tune", "multilayer"):
                if family == "multilayer" and dataset not in protocol[family]["datasets"]:
                    continue
                for variant in protocol[family]["variants"]:
                    result.append({"id": f"{dataset}/{seed}/{family}/{variant}", "dataset": dataset,
                                   "seed": seed, "family": family, "variant": variant, "group": group})
            group += 1
    return result


def pretrained_embedding(study, dataset):
    if dataset != "conll":
        path = Path(study["models"]["dnabert"])
        state = torch.load(path / "pytorch_model.bin", weights_only=True, mmap=True, map_location="cpu")
        return state["bert.embeddings.word_embeddings.weight"].float().clone(), 0
    root = Path(study["models"]["llama"])
    index = json.loads((root / "model.safetensors.index.json").read_text())
    key = "model.embed_tokens.weight"
    with safe_open(root / index["weight_map"][key], framework="pt", device="cpu") as f:
        weight = f.get_tensor(key).float().clone()
    # Match the pretrained Llama embedding module, whose padding_idx is config.pad_token_id.
    return weight, json.loads((root / "config.json").read_text()).get("pad_token_id")


def single_model(study, dataset, variant):
    weight, padding_idx = pretrained_embedding(study, dataset)
    embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=padding_idx)
    is_tokens = dataset == "conll"
    width = weight.shape[1]
    args = {"d_model": width, "nhead": 32 if is_tokens else 12, "d_ff": 4 * width,
            "num_emb": weight.shape[0], "max_seq_len": 64 if is_tokens else 256,
            "num_classes": 47 if is_tokens else 2, "embedding_layer": embedding}
    if variant == "fn":
        cls = fn.FrobeniuslyNormalizedTransformerForTokenClassification if is_tokens else fn.FrobeniuslyNormalizedTransformerForSequenceClassification
        return cls(**args, apply_attention_fn=True, apply_ffn_fn=True)
    cls = sn.SpectrallyNormalizedTransformerForTokenClassification if is_tokens else sn.SpectrallyNormalizedTransformerForSequenceClassification
    return cls(**args, apply_attention_sn=variant in ("sn", "sn_attention"),
               apply_ffn_sn=variant in ("sn", "sn_ffn"))


def baseline_multilayer(study):
    config = BertConfig.from_pretrained(study["models"]["dnabert"], local_files_only=True)
    config.attention_probs_dropout_prob = 0.1
    model, info = AutoModelForSequenceClassification.from_pretrained(
        study["models"]["dnabert"], config=config, trust_remote_code=True,
        local_files_only=True, output_loading_info=True)
    expected = {"bert.pooler.dense.weight", "bert.pooler.dense.bias", "classifier.weight", "classifier.bias"}
    if set(info["missing_keys"]) != expected or info["mismatched_keys"] or info["error_msgs"]:
        raise RuntimeError(f"Unexpected pretrained loading coverage: {info}")
    if any(not key.startswith("cls.") for key in info["unexpected_keys"]):
        raise RuntimeError(f"Unexpected pretrained keys: {info['unexpected_keys']}")
    return model


def verify_copied_weights(model, source):
    target = model.state_dict()
    fixed = {f"{name}.{key}" for name, module in model.named_modules() if isinstance(module, UnitNormLayerNorm)
             for key in ("weight", "bias")}
    checked, intentional = 0, []
    for name, value in source.items():
        if name in fixed:
            intentional.append(name)
            continue
        parts = [(name, value)]
        if ".Wqkv." in name and name not in target:
            parts = [(name.replace(".Wqkv.", ".W" + kind + "."), part) for kind, part in zip("qkv", value.chunk(3, dim=0))]
        for key, part in parts:
            if key not in target and key.endswith(".weight"):
                key = key[:-7] + ".parametrizations.weight.original"
            if key not in target or not torch.equal(target[key], part):
                raise AssertionError(f"Initial raw weight mismatch: {key}")
            checked += part.numel()
    return {"verified_source_elements": checked, "intentional_fixed_layernorm_buffers": intentional}


def build_model(study, root, job):
    seed_everything(job["seed"])
    dataset, family, variant = job["dataset"], job["family"], job["variant"]
    init = root / "initial" / dataset / str(job["seed"]) / ("multilayer.pt" if family == "multilayer" else "single.pt")
    init.parent.mkdir(parents=True, exist_ok=True)
    if not init.exists():
        base = baseline_multilayer(study) if family == "multilayer" else single_model(study, dataset, "vanilla")
        save_checkpoint(init, base.state_dict())
        del base
    parent = None
    if family == "fine_tune":
        parent = root / "runs" / dataset / str(job["seed"]) / "single/vanilla"
        if not (parent / "selected.json").exists():
            raise RuntimeError(f"Vanilla parent has not finished: {parent}")
        source_path = parent / "best.pt"
        source = torch.load(source_path, map_location="cpu", weights_only=True, mmap=True)["model"]
    else:
        source_path = init
        source = torch.load(init, map_location="cpu", weights_only=True, mmap=True)
    seed_everything(job["seed"])
    if family == "multilayer":
        if variant == "vanilla":
            model = baseline_multilayer(study)
            model.load_state_dict(source, strict=True)
        else:
            config = BertConfig.from_pretrained(study["models"]["dnabert"], local_files_only=True)
            config.attention_probs_dropout_prob = 0.1
            maximum = study["protocol"]["multilayer"][f"{dataset}_fn_max_gamma"] if variant == "fn" else 1
            norm = FrobeniuslyNormalizedBertConfig.from_bert_config(
                config, apply_attn_fn=True, apply_ffn_fn=True, no_layer_norm=False, max_gamma=maximum)
            model = FrobeniuslyNormalizedBertForSequenceClassification(norm)
            load_unparametrized_checkpoint(model, source)
    else:
        model = single_model(study, dataset, variant)
        load_unparametrized_checkpoint(model, source)
    coverage = verify_copied_weights(model, source)
    coverage.update(source_path=str(source_path), source_sha256=file_sha(source_path),
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad))
    if str(source_path) in study.get("initial_sha256", {}) and coverage["source_sha256"] != study["initial_sha256"][str(source_path)]:
        raise RuntimeError(f"Frozen initialization was modified: {source_path}")
    return model, coverage


def batches(data, size, order=None):
    if order is None:
        order = torch.arange(len(data["input_ids"]))
    for start in range(0, len(order), size):
        ix = order[start:start + size]
        yield ix, {k: v[ix].cuda(non_blocking=False) for k, v in data.items()}


def forward(model, batch, family):
    if family == "multilayer":
        return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    return model(batch["input_ids"], key_padding_mask=batch["attention_mask"] == 0)


def loss_sum(logits, labels):
    return nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                       ignore_index=-100, reduction="sum")


def score(labels, predictions, token_task):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    keep = labels != -100
    actual, predicted = labels[keep], predictions[keep]
    if not len(actual):
        raise ValueError("Evaluation has no labeled elements")
    return {"accuracy": float(accuracy_score(actual, predicted)),
            "f1": float(f1_score(actual, predicted, average="macro" if token_task else "binary", zero_division=0)),
            "labeled_elements": len(actual)}


@torch.inference_mode()
def evaluate(model, data, job, size, return_logits=False):
    model.eval()
    predictions, scores, total_loss, count = [], [], 0.0, 0
    for _, batch in batches(data, size):
        logits = forward(model, batch, job["family"])
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation logits")
        total_loss += loss_sum(logits, batch["labels"]).item()
        count += (batch["labels"] != -100).sum().item()
        predictions.append(logits.argmax(-1).cpu())
        if return_logits:
            scores.append(logits.cpu())
    predictions = torch.cat(predictions)
    result = score(data["labels"].numpy(), predictions.numpy(), job["dataset"] == "conll")
    result["loss"] = total_loss / count
    return result, predictions, torch.cat(scores) if return_logits else None


def optimizer_for(model, config, family):
    if family == "multilayer":
        no_decay = {id(p) for m in model.modules() if isinstance(m, (nn.LayerNorm, UnitNormLayerNorm)) for p in m.parameters()}
        decay, other = [], []
        for name, parameter in model.named_parameters():
            (other if name.endswith(".bias") or id(parameter) in no_decay else decay).append(parameter)
        parameters = [{"params": decay, "weight_decay": config["weight_decay"]}, {"params": other, "weight_decay": 0.0}]
    else:
        parameters = model.parameters()
    return torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=config["weight_decay"],
                             betas=(0.9, 0.999), eps=1e-8, foreach=False)


def train_job(study, root, job):
    folder = root / "runs" / job["id"]
    if (folder / "selected.json").exists():
        locked = json.loads((folder / "selected.json").read_text())
        if locked["checkpoint_sha256"] != file_sha(folder / "best.pt"):
            raise RuntimeError(f"Selected checkpoint was modified: {folder}")
        return
    folder.mkdir(parents=True, exist_ok=True)
    config = settings(study["protocol"], job["dataset"], job["family"])
    model, coverage = build_model(study, root, job)
    model.cuda()
    train = torch.load(root / "data" / job["dataset"] / "train.pt", weights_only=True)
    validation = torch.load(root / "data" / job["dataset"] / "validation.pt", weights_only=True)
    optimizer = optimizer_for(model, config, job["family"])
    steps_per_epoch = math.ceil(len(train["input_ids"]) / config["batch_size"])
    if job["family"] == "multilayer":
        steps = steps_per_epoch * config["epochs"]
        scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(steps * config["warmup_ratio"]), steps)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    seed_everything(job["seed"])
    history, best, best_epoch, stale, first_epoch = [], -math.inf, None, 0, 1
    latest = folder / "latest.pt"
    if latest.exists():
        # Local trusted checkpoint includes Python and NumPy RNG state.
        state = torch.load(latest, weights_only=False, map_location="cpu")
        if state["study_sha256"] != study["study_sha256"]:
            raise RuntimeError("Refusing to resume against a changed study")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best, best_epoch, stale = state["history"], state["best"], state["best_epoch"], state["stale"]
        first_epoch = state["epoch"] + 1
        restore_rng(state["rng"])
        del state
    else:
        initial, _, _ = evaluate(model, validation, job, config["eval_batch_size"])
        history.append({"epoch": 0, "validation": initial})
    write_json(folder / "run.json", {"job": job, "settings": config, "study_sha256": study["study_sha256"],
               "initialization": coverage, "gpu": torch.cuda.get_device_name(), "pid": os.getpid(),
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "batch_permutation": "torch.randperm(n, generator=manual_seed(seed*1000003+epoch))"})
    for epoch in range(first_epoch, config["epochs"] + 1):
        if config["patience"] is not None and stale >= config["patience"]:
            break
        started = time.monotonic()
        model.train()
        order = epoch_order(len(train["input_ids"]), job["seed"], epoch)
        loss_total, labeled, max_grad, learning_rate = 0.0, 0, 0.0, optimizer.param_groups[0]["lr"]
        for _, batch in batches(train, config["batch_size"], order):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = forward(model, batch, job["family"])
                count = (batch["labels"] != -100).sum()
                loss = loss_sum(logits, batch["labels"]) / count
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss: {job['id']} epoch {epoch}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"] or math.inf,
                                                 error_if_nonfinite=True, foreach=False)
            max_grad = max(max_grad, grad.item())
            optimizer.step()
            if job["family"] == "multilayer":
                scheduler.step()
            loss_total += loss.item() * count.item()
            labeled += count.item()
        validation_metrics, _, _ = evaluate(model, validation, job, config["eval_batch_size"])
        if job["family"] != "multilayer":
            scheduler.step()
        metric = validation_metrics[config["metric"]]
        if metric > best:
            best, best_epoch, stale = metric, epoch, 0
            save_checkpoint(folder / "best.pt", {"model": model.state_dict(), "epoch": epoch,
                            "validation": validation_metrics, "study_sha256": study["study_sha256"]})
        else:
            stale += 1
        history.append({"epoch": epoch, "validation": validation_metrics, "train_loss": loss_total / labeled,
                        "max_gradient_norm": max_grad, "learning_rate": learning_rate,
                        "seconds": time.monotonic() - started,
                        "sample_order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest()})
        save_checkpoint(latest, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "rng": capture_rng(), "epoch": epoch,
                        "history": history, "best": best, "best_epoch": best_epoch, "stale": stale,
                        "study_sha256": study["study_sha256"]})
        write_json(folder / "history.json", history)
        print(json.dumps({"run": job["id"], **history[-1], "best_epoch": best_epoch}), flush=True)
    selected = torch.load(folder / "best.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(selected["model"], strict=True)
    metrics, predictions, _ = evaluate(model, validation, job, config["eval_batch_size"])
    if metrics != selected["validation"]:
        raise AssertionError(f"Restored validation results differ: {metrics}, {selected['validation']}")
    np.savez_compressed(folder / "validation_predictions.npz", labels=validation["labels"].numpy(), predictions=predictions.numpy())
    write_json(folder / "selected.json", {"job": job, "best_epoch": best_epoch,
               "trained_epochs": history[-1]["epoch"], "validation": metrics,
               "checkpoint_sha256": file_sha(folder / "best.pt"), "study_sha256": study["study_sha256"],
               "stopped_early": history[-1]["epoch"] < config["epochs"], "test_evaluated": False})
    print(json.dumps({"selected": job["id"], "best_epoch": best_epoch, "validation": metrics}), flush=True)
    del model, optimizer, scheduler, selected
    gc.collect()
    torch.cuda.empty_cache()


def preflight(study, root):
    results = []
    for dataset in study["protocol"]["datasets"]:
        for family, variants in (("single", ["vanilla", "sn", "fn"]), ("multilayer", ["vanilla", "fn", "subnorm"])):
            if family == "multilayer" and dataset == "conll":
                continue
            for variant in variants:
                job = {"dataset": dataset, "family": family, "variant": variant, "seed": 42}
                model, coverage = build_model(study, root, job)
                model.cuda()
                data = torch.load(root / "data" / dataset / "validation.pt", weights_only=True)
                data = {k: v[:4] for k, v in data.items()}
                _, predictions_a, logits_a = evaluate(model, data, job, 4, return_logits=True)
                _, predictions_b, logits_b = evaluate(model, data, job, 1, return_logits=True)
                torch.testing.assert_close(logits_a, logits_b, atol=2e-4, rtol=2e-4)
                torch.testing.assert_close(predictions_a, predictions_b, rtol=0, atol=0)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
                model.train()
                training = torch.load(root / "data" / dataset / "train.pt", weights_only=True)
                batch_size = settings(study["protocol"], dataset, family)["batch_size"]
                batch = {k: v[:batch_size].cuda() for k, v in training.items()}
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                seed_everything(42)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = forward(model, batch, family)
                        loss = loss_sum(logits, batch["labels"]) / (batch["labels"] != -100).sum()
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf, error_if_nonfinite=True)
                    optimizer.step()
                results.append({**job, "validation_batch_invariance": True, "coverage": coverage,
                                "loss": loss.item(), "gradient_norm": grad.item(), "training_batch_size": batch_size,
                                "two_steps_seconds": time.monotonic() - started,
                                "peak_cuda_bytes": torch.cuda.max_memory_allocated()})
                print(json.dumps(results[-1]), flush=True)
                del model, optimizer, data, training, batch, logits, loss
                gc.collect()
                torch.cuda.empty_cache()
    write_json(root / "preflight.json", results)


@torch.inference_mode()
def norm_audit(model):
    model.eval()
    records = []
    for name, module in model.named_modules():
        if not hasattr(module, "parametrizations") or "weight" not in module.parametrizations:
            continue
        weight = module.weight.float()
        record = {"module": name, "shape": list(weight.shape),
                  "frobenius": torch.linalg.vector_norm(weight).item()}
        transform = module.parametrizations.weight[0]
        if hasattr(transform, "_u"):
            # Independently measure the effective weight; power iteration is not a certified hard cap.
            record["spectral_svd"] = torch.linalg.matrix_norm(weight, ord=2).item()
            record["normalization"] = "PyTorch spectral parametrization (one power iteration per training forward)"
        else:
            record["normalization"] = type(transform).__name__
        records.append(record)
    return records


def finalize(study, root, worker, workers):
    locks = {}
    for job in study["jobs"]:
        folder = root / "runs" / job["id"]
        selected = folder / "selected.json"
        if not selected.exists():
            raise RuntimeError(f"Test evaluation is locked until all runs finish; missing {job['id']}")
        record = json.loads(selected.read_text())
        if record["study_sha256"] != study["study_sha256"] or file_sha(folder / "best.pt") != record["checkpoint_sha256"]:
            raise RuntimeError(f"Selected checkpoint does not match its frozen record: {job['id']}")
        locks[job["id"]] = record["checkpoint_sha256"]
    lock_path = root / "selection_lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text())["checkpoints"] != locks:
            raise RuntimeError("Selected checkpoints changed after the test lock")
    else:
        write_json(lock_path, {"study_sha256": study["study_sha256"], "checkpoints": locks,
                              "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    for job in study["jobs"]:
        if job["group"] % workers != worker:
            continue
        folder = root / "runs" / job["id"]
        if (folder / "test.json").exists():
            continue
        model, _ = build_model(study, root, job)
        state = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        model.cuda().eval()
        data = torch.load(root / "data" / job["dataset"] / "test.pt", weights_only=True)
        config = settings(study["protocol"], job["dataset"], job["family"])
        metrics, predictions, logits = evaluate(model, data, job, config["eval_batch_size"], return_logits=True)
        path = folder / "test_predictions.npz"
        np.savez_compressed(path, labels=data["labels"].numpy(), predictions=predictions.numpy(), logits=logits.numpy())
        audit = norm_audit(model)
        write_json(folder / "test.json", {"job": job, "metrics": metrics, "norms": audit,
                   "selected_epoch": state["epoch"], "checkpoint_sha256": locks[job["id"]],
                   "predictions_sha256": file_sha(path), "study_sha256": study["study_sha256"]})
        print(json.dumps({"test": job["id"], "metrics": metrics}), flush=True)
        del model, state, data, logits, predictions
        gc.collect()
        torch.cuda.empty_cache()


def summarize(study, root):
    from scipy.stats import t

    rows, records = [], {}
    for job in study["jobs"]:
        path = root / "runs" / job["id"] / "test.json"
        if not path.exists():
            raise RuntimeError(f"Cannot report an incomplete study: {job['id']}")
        record = json.loads(path.read_text())
        if record["study_sha256"] != study["study_sha256"]:
            raise RuntimeError("Test result belongs to another study")
        records[job["id"]] = record
    for dataset in study["protocol"]["datasets"]:
        for family in ("single", "fine_tune", "multilayer"):
            if family == "multilayer" and dataset == "conll":
                continue
            for variant in study["protocol"][family]["variants"]:
                for metric in ("accuracy", "f1"):
                    values = [records[f"{dataset}/{seed}/{family}/{variant}"]["metrics"][metric] for seed in study["protocol"]["seeds"]]
                    baseline_family = "single" if family == "fine_tune" else family
                    baselines = [records[f"{dataset}/{seed}/{baseline_family}/vanilla"]["metrics"][metric] for seed in study["protocol"]["seeds"]]
                    delta = np.asarray(values) - np.asarray(baselines)
                    n = len(values)
                    row = {"dataset": dataset, "family": family, "variant": variant, "metric": metric,
                           "seeds": study["protocol"]["seeds"], "values": values,
                           "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)),
                           "mean_95ci_halfwidth": float(t.ppf(0.975, n-1) * np.std(values, ddof=1) / np.sqrt(n)),
                           "paired_delta_from_vanilla": delta.tolist(), "paired_delta_mean": float(delta.mean()),
                           "paired_delta_95ci_halfwidth": float(t.ppf(0.975, n-1) * delta.std(ddof=1) / np.sqrt(n))}
                    rows.append(row)
    write_json(root / "summary.json", {"study_sha256": study["study_sha256"], "rows": rows,
               "uncertainty": "Student-t intervals across five paired training seeds on one fixed test set; not uncertainty over data splits or populations; no multiplicity-adjusted significance claims.",
               "all_runs": records})
    lines = ["Corrected legacy experiments: all seeds", "", "| Dataset | Family | Variant | Accuracy (mean ± sample SD) | F1 (mean ± sample SD) |", "|---|---|---|---:|---:|"]
    for a, f in zip(rows[::2], rows[1::2]):
        lines.append(f"| {a['dataset']} | {a['family']} | {a['variant']} | {100*a['mean']:.3f} ± {100*a['sample_sd']:.3f}% | {100*f['mean']:.3f} ± {100*f['sample_sd']:.3f}% |")
    lines += ["", "F1 is binary positive-class F1 for DNA and macro F1 for CoNLL POS.",
              "Genomic validation and test protocol differs from the historical table; differences cannot be attributed solely to the code repair.",
              "The multilayer comparison changes LayerNorm, residual scaling and attention temperature as well as weight normalization."]
    (root / "summary.md").write_text("\n".join(lines) + "\n")


def freeze(root, protocol_path, sources_path, llama_path):
    if (root / "study.json").exists():
        raise FileExistsError("Study already frozen; create a new study for protocol changes")
    repo = Path(__file__).resolve().parents[2]
    protocol = json.loads(protocol_path.read_text())
    sources = json.loads(sources_path.read_text())
    snapshot = root / "source"
    for relative in ("src", "experiments/rigorous_rerun"):
        shutil.copytree(repo / relative, snapshot / relative, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
    shutil.copy2(protocol_path, snapshot / "protocol.json")
    code = {str(p.relative_to(snapshot)): file_sha(p) for p in sorted(snapshot.rglob("*")) if p.is_file()}
    data = {name: json.loads((root / "data" / name / "manifest.json").read_text()) for name in protocol["datasets"]}
    model_paths = {"dnabert": sources["zhihan1996/DNABERT-2-117M"]["snapshot_path"], "llama": llama_path}
    model_hashes = {}
    for model_path in model_paths.values():
        for p in sorted(Path(model_path).iterdir()):
            if p.is_file() and p.suffix in (".bin", ".safetensors", ".json", ".py", ".model"):
                model_hashes[str(p)] = file_sha(p)
    study = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "protocol": protocol, "sources": sources, "models": model_paths, "model_sha256": model_hashes,
             "source_sha256": code, "data": data, "jobs": jobs(protocol),
             "environment": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "numpy", "pandas", "scikit-learn", "safetensors")}}
    study["preflight_sha256"] = file_sha(root / "preflight.json")
    preflight_rows = json.loads((root / "preflight.json").read_text())
    expected = {(j["dataset"], j["family"], j["variant"]) for j in study["jobs"] if j["family"] != "fine_tune"}
    actual = {(r["dataset"], r["family"], r["variant"]) for r in preflight_rows if r["validation_batch_invariance"] and r.get("training_batch_size") == settings(protocol, r["dataset"], r["family"])["batch_size"]}
    if actual != expected:
        raise RuntimeError("Full training-batch preflight is incomplete")
    study["initial_sha256"] = {str(p): file_sha(p) for p in sorted((root / "initial").rglob("*.pt"))}
    if (root / "startup_amendment.json").exists():
        study["startup_amendment"] = json.loads((root / "startup_amendment.json").read_text())
    study["study_sha256"] = digest(study)
    write_json(root / "study.json", study)
    print(json.dumps({"study_sha256": study["study_sha256"], "jobs": len(study["jobs"]), "root": str(root)}), flush=True)


def verify_study(study, root):
    recorded = study["study_sha256"]
    if digest({k: v for k, v in study.items() if k != "study_sha256"}) != recorded:
        raise RuntimeError("Frozen study manifest was modified")
    for relative, expected in study["source_sha256"].items():
        if file_sha(root / "source" / relative) != expected:
            raise RuntimeError(f"Frozen source changed: {relative}")
    source = Path(__file__).resolve()
    if root / "source" not in source.parents:
        raise RuntimeError("Execute the frozen runner using PYTHONPATH=<study>/source:<study>/source/src")
    for name, manifest in study["data"].items():
        for split in ("train", "validation", "test"):
            if file_sha(root / "data" / name / f"{split}.pt") != manifest["splits"][split]["tensor_sha256"]:
                raise RuntimeError(f"Prepared data changed: {name}/{split}")
            if file_sha(root / "data" / name / f"{split}_rows.json") != manifest["splits"][split]["row_manifest_sha256"]:
                raise RuntimeError(f"Prepared sample identities changed: {name}/{split}")
    if file_sha(root / "preflight.json") != study["preflight_sha256"]:
        raise RuntimeError("Preflight record changed after freezing")
    for path, expected in study["model_sha256"].items():
        if file_sha(path) != expected:
            raise RuntimeError(f"Pinned model/tokenizer artifact changed: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "preflight", "queue", "status", "finalize", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/configs/training/rigorous_rerun.json"))
    parser.add_argument("--sources", type=Path, default=Path(".cache/rigorous_rerun/sources.json"))
    parser.add_argument("--llama-path")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.root, args.protocol, args.sources, args.llama_path)
        return
    study = json.loads((args.root / "study.json").read_text())
    if args.command == "status":
        completed = []
        active = []
        for job in study["jobs"]:
            folder = args.root / "runs" / job["id"]
            if (folder / "selected.json").exists():
                completed.append(job["id"])
            elif (folder / "history.json").exists():
                history = json.loads((folder / "history.json").read_text())
                active.append({"run": job["id"], "last_epoch": history[-1]})
        print(json.dumps({"total": len(study["jobs"]), "completed": completed, "active": active}, indent=2))
        return
    if args.command == "summarize":
        summarize(study, args.root)
        return
    configure()
    verify_study(study, args.root)
    if args.command == "preflight":
        preflight(study, args.root)
        return
    if args.command == "finalize":
        finalize(study, args.root, args.worker, args.workers)
        return
    if not (args.root / "preflight.json").exists():
        raise RuntimeError("Full-shape preflight must pass before training")
    for job in study["jobs"]:
        if job["group"] % args.workers != args.worker:
            continue
        try:
            train_job(study, args.root, job)
        except Exception as error:
            write_json(args.root / "failures" / f"worker{args.worker}.json", {"job": job,
                       "error": repr(error), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            raise
    write_json(args.root / f"worker{args.worker}_complete.json", {"study_sha256": study["study_sha256"]})


if __name__ == "__main__":
    main()
