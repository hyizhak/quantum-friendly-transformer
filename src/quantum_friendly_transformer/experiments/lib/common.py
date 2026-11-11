import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


GLUE_TEXT_COLUMNS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


@dataclass(frozen=True)
class Variant:
    raw: str
    kind: str
    tau: float | None = None
    recovery: bool = False


def parse_variant(spec: str) -> Variant:
    spec = spec.strip().lower()
    recovery = spec.endswith("+recovery")
    base = spec.removesuffix("+recovery")
    if base == "baseline":
        return Variant(raw=spec, kind="baseline", recovery=recovery)
    if ":" not in base:
        raise ValueError(f"variant must be baseline, sn:<tau>, or fn:<tau>: {spec}")
    kind, tau_s = base.split(":", 1)
    if kind not in {"sn", "fn"}:
        raise ValueError(f"unsupported variant kind {kind!r}")
    return Variant(raw=spec, kind=kind, tau=float(tau_s), recovery=recovery)


def ensure_parent(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def write_json(path: str | os.PathLike[str], payload: dict) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def metadata_path(output_path: str | os.PathLike[str]) -> Path:
    output_path = Path(output_path)
    return output_path.with_suffix(output_path.suffix + ".metadata.json")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def now_metadata() -> dict:
    return {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script_commit": git_commit(),
    }


def package_versions() -> dict:
    versions = {}
    for name in ["torch", "transformers", "datasets", "pandas", "pyarrow", "numpy"]:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "not-installed"
    return versions


def glue_split_name(task: str, split: str) -> str:
    if task == "mnli" and split == "validation":
        return "validation_matched"
    return split


def glue_num_labels(task: str) -> int:
    if task == "stsb":
        return 1
    if task == "mnli":
        return 3
    return 2


def compute_classification_metrics(task: str, predictions, labels) -> dict:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    if task == "stsb":
        from scipy.stats import pearsonr, spearmanr

        preds = np.asarray(predictions, dtype=float)
        labs = np.asarray(labels, dtype=float)
        return {
            "pearson": float(pearsonr(preds, labs).statistic),
            "spearman": float(spearmanr(preds, labs).statistic),
        }

    preds = np.asarray(predictions)
    labs = np.asarray(labels)
    out = {"accuracy": float(accuracy_score(labs, preds))}
    if task in {"mrpc", "qqp"}:
        out["f1"] = float(f1_score(labs, preds))
    if task == "cola":
        out["matthews"] = float(matthews_corrcoef(labs, preds))
    return out



from quantum_friendly_transformer.norm_transformer.frobenius_cap import (
    OutputFrobeniusCapWrapper,
    apply_learnable_frobenius_caps,
    collect_learnable_cap_stats,
    collect_weight_stats,
    estimate_module_frobenius_norm,
    is_transformer_target_module_name,
    is_transformer_target_parameter_name,
    learnable_frobenius_cap,
    target_named_modules,
    target_weight_modules,
    target_weight_parameters,
)

# Backward-compatible aliases for older experiment scripts. The implementation
# lives in quantum_friendly_transformer.norm_transformer.frobenius_cap.
_is_target_weight_module_name = is_transformer_target_module_name
_is_target_weight_parameter_name = is_transformer_target_parameter_name
make_learnable_frobenius_cap = learnable_frobenius_cap

def spectral_norm_power_iteration(torch, weight, iters: int = 2):
    matrix = weight.float()
    if matrix.numel() == 0:
        return torch.tensor(float("nan"), device=matrix.device)
    # Deterministic initial vector keeps projections reproducible without storing per-weight state.
    v = torch.ones(matrix.shape[1], device=matrix.device, dtype=matrix.dtype)
    v = v / (torch.linalg.vector_norm(v) + 1e-12)
    for _ in range(max(1, iters)):
        u = matrix @ v
        u = u / (torch.linalg.vector_norm(u) + 1e-12)
        v = matrix.transpose(0, 1) @ u
        v = v / (torch.linalg.vector_norm(v) + 1e-12)
    return torch.dot(u, matrix @ v).abs()


def project_model_weights(model, kind: str, tau: float) -> int:
    import torch

    if kind not in {"sn", "fn"}:
        return 0
    projected = 0
    with torch.no_grad():
        for _, param in target_weight_parameters(model):
            weight = param.data.float()
            if kind == "fn":
                norm = torch.linalg.vector_norm(weight)
            else:
                norm = spectral_norm_power_iteration(torch, weight, iters=2)
            if torch.isfinite(norm) and norm.item() > tau:
                param.data.mul_(float(tau / norm.item()))
                projected += 1
    return projected


def model_config_summary(model) -> dict:
    config = model.config
    model_type = getattr(config, "model_type", "unknown")
    if model_type == "gemma3":
        config = getattr(config, "text_config", None) or getattr(getattr(model, "language_model", None), "config", config)
        model_type = getattr(config, "model_type", model_type)
    layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "n_layer", None)
        or getattr(config, "num_layers", None)
    )
    hidden = (
        getattr(config, "hidden_size", None)
        or getattr(config, "n_embd", None)
        or getattr(config, "d_model", None)
    )
    heads = getattr(config, "num_attention_heads", None) or getattr(config, "n_head", None)
    rms_norm_types = {"llama", "mistral", "qwen2", "qwen3", "phi3", "gemma", "gemma2", "gemma3_text", "smollm3"}
    causal_types = {
        "gpt2",
        "gpt_bigcode",
        "gpt_neo",
        "gpt_neox",
        "opt",
        "bloom",
        "llama",
        "mistral",
        "qwen2",
        "qwen3",
        "olmo",
        "phi",
        "phi3",
        "gemma",
        "gemma2",
        "gemma3_text",
        "smollm3",
    }
    norm_type = "RMSNorm" if model_type in rms_norm_types else "LayerNorm"
    attention_type = "causal" if model_type in causal_types else "bidirectional"
    return {
        "model_type": model_type,
        "num_layers": layers,
        "hidden_dim": hidden,
        "num_heads": heads,
        "norm_type": norm_type,
        "attention_type": attention_type,
    }


def finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None
