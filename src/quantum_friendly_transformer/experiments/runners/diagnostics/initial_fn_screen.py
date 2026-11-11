#!/usr/bin/env python
"""Screen d-sized causal LMs by initial target-weight Frobenius norms."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import time
import traceback
from pathlib import Path

from quantum_friendly_transformer.norm_transformer.frobenius_cap import target_weight_modules

from quantum_friendly_transformer.experiments.lib.common import (
    now_metadata,
    package_versions,
    write_json,
)


DEFAULT_MODELS = [
    "google/gemma-2b",
    "google/gemma-2b-it",
    "stabilityai/stablelm-2-1_6b",
    "Qwen/Qwen1.5-1.8B",
    "Qwen/Qwen3-1.7B",
    "microsoft/phi-1",
    "microsoft/phi-1_5",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/gpt-neo-1.3B",
    "facebook/galactica-1.3b",
    "facebook/opt-1.3b",
    "facebook/opt-iml-1.3b",
    "tiiuae/falcon-rw-1b",
    "bigscience/bloom-1b7",
    "bigscience/bloomz-1b7",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "unsloth/Llama-3.2-1B-Instruct",
    "allenai/OLMo-1B-hf",
    "allenai/OLMo-1B-0724-hf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--models-file", default=None)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--include-other-sizes", action="store_true")
    parser.add_argument("--cap", type=float, default=24.0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", default=None)
    return parser.parse_args()


def unique_models(args: argparse.Namespace) -> list[str]:
    models = list(args.models)
    if args.models_file:
        path = Path(args.models_file)
        models.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    seen = set()
    out = []
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        out.append(model)
    return out


def config_hidden_size(config) -> int | None:
    for attr in ("hidden_size", "n_embd", "d_model", "word_embed_proj_dim"):
        value = getattr(config, attr, None)
        if isinstance(value, int):
            return value
    return None


def torch_dtype(torch, dtype: str):
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    return torch.float32


def pick_device(torch, requested: str):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(ordered[lower])
    frac = pos - lower
    return float(ordered[lower] * (1.0 - frac) + ordered[upper] * frac)


def target_group(name: str) -> str:
    lowered = name.lower()
    if (
        ".self_attn." in lowered
        or ".attention." in lowered
        or ".attn." in lowered
        or ".self_attention." in lowered
    ):
        return "attention"
    if ".mlp." in lowered or ".fc1" in lowered or ".fc2" in lowered or "intermediate.dense" in lowered:
        return "mlp"
    return "other"


def summarize_norms(norms: list[float], cap: float) -> dict:
    if not norms:
        return {}
    scales = [min(1.0, cap / max(norm, 1e-12)) for norm in norms]
    overs = [max(0.0, norm - cap) for norm in norms]
    le_cap = sum(1 for norm in norms if norm <= cap)
    return {
        "fn_min": min(norms),
        "fn_p10": quantile(norms, 0.10),
        "fn_p25": quantile(norms, 0.25),
        "fn_median": quantile(norms, 0.50),
        "fn_mean": sum(norms) / len(norms),
        "fn_p75": quantile(norms, 0.75),
        "fn_p90": quantile(norms, 0.90),
        "fn_p95": quantile(norms, 0.95),
        "fn_max": max(norms),
        "le_cap_count": le_cap,
        "le_cap_frac": le_cap / len(norms),
        "cap_scale_min": min(scales),
        "cap_scale_p10": quantile(scales, 0.10),
        "cap_scale_median": quantile(scales, 0.50),
        "cap_scale_mean": sum(scales) / len(scales),
        "over_cap_median": quantile(overs, 0.50),
        "over_cap_p90": quantile(overs, 0.90),
        "over_cap_max": max(overs),
    }


def weight_rows(torch, model, cap: float) -> tuple[list[dict], dict]:
    rows = []
    group_norms = {"attention": [], "mlp": [], "other": []}
    with torch.no_grad():
        for name, module in target_weight_modules(model):
            weight = module.weight.detach()
            norm = float(torch.linalg.vector_norm(weight.float()).cpu().item())
            group = target_group(name)
            group_norms[group].append(norm)
            rows.append(
                {
                    "name": f"{name}.weight",
                    "shape": list(weight.shape),
                    "group": group,
                    "frobenius_norm": norm,
                    "cap_scale": min(1.0, cap / max(norm, 1e-12)),
                    "over_cap": max(0.0, norm - cap),
                }
            )
    groups = {group: summarize_norms(norms, cap) for group, norms in group_norms.items() if norms}
    return rows, groups


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "status",
        "model",
        "model_type",
        "hidden_size",
        "target_count",
        "fn_median",
        "fn_p90",
        "fn_p95",
        "fn_max",
        "le_cap_frac",
        "cap_scale_median",
        "cap_scale_p10",
        "cap_scale_min",
        "over_cap_p90",
        "over_cap_max",
        "elapsed_sec",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def ranking_key(row: dict) -> tuple[float, float, float, float]:
    if row.get("status") != "ok":
        return (float("inf"), float("inf"), float("inf"), float("inf"))
    return (
        float(row.get("fn_p90") or float("inf")),
        float(row.get("fn_max") or float("inf")),
        -float(row.get("cap_scale_median") or 0.0),
        -float(row.get("le_cap_frac") or 0.0),
    )


def release(torch, model=None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    args = parse_args()
    models = unique_models(args)
    device = pick_device(torch, args.device)
    dtype = torch_dtype(torch, args.dtype)
    csv_output = args.csv_output or str(Path(args.output).with_suffix(".csv"))

    payload = {
        "metadata": {
            **now_metadata(),
            "versions": package_versions(),
            "screen": "initial_target_weight_frobenius",
            "cap": args.cap,
            "hidden_size": args.hidden_size,
            "dtype": args.dtype,
            "device": str(device),
            "models": models,
        },
        "results": [],
        "ranking": [],
    }

    common_kwargs = {
        "cache_dir": args.cache_dir,
        "local_files_only": args.offline,
        "trust_remote_code": args.trust_remote_code,
    }

    for index, model_name in enumerate(models, start=1):
        start = time.time()
        print(f"[{index}/{len(models)}] loading config {model_name}", flush=True)
        result = {"model": model_name, "status": "pending"}
        model = None
        try:
            config = AutoConfig.from_pretrained(model_name, **common_kwargs)
            hidden = config_hidden_size(config)
            result.update(
                {
                    "model_type": getattr(config, "model_type", None),
                    "architectures": getattr(config, "architectures", None),
                    "hidden_size": hidden,
                    "num_hidden_layers": getattr(config, "num_hidden_layers", None),
                    "num_attention_heads": getattr(config, "num_attention_heads", None),
                    "intermediate_size": getattr(config, "intermediate_size", None),
                }
            )
            if hidden != args.hidden_size and not args.include_other_sizes:
                result["status"] = "skipped_hidden_size"
                result["error"] = f"hidden_size={hidden}, expected={args.hidden_size}"
                print(f"  skipped: {result['error']}", flush=True)
            else:
                print(f"  loading weights as {args.dtype} on {device}", flush=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    **common_kwargs,
                )
                model.eval()
                if device.type != "cpu":
                    model.to(device)
                rows, groups = weight_rows(torch, model, args.cap)
                norms = [row["frobenius_norm"] for row in rows]
                result.update(summarize_norms(norms, args.cap))
                result.update(
                    {
                        "status": "ok" if rows else "no_targets",
                        "target_count": len(rows),
                        "groups": groups,
                        "largest_targets": sorted(rows, key=lambda row: row["frobenius_norm"], reverse=True)[:10],
                        "smallest_targets": sorted(rows, key=lambda row: row["frobenius_norm"])[:10],
                    }
                )
                print(
                    "  targets={target_count} p50={fn_median:.3f} p90={fn_p90:.3f} "
                    "max={fn_max:.3f} scale_p50={cap_scale_median:.3f}".format(**result),
                    flush=True,
                )
        except Exception as exc:
            result.update(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"  error: {result['error']}", flush=True)
        finally:
            result["elapsed_sec"] = time.time() - start
            payload["results"].append(result)
            ok_rows = [row for row in payload["results"] if row.get("status") == "ok"]
            ranked = sorted(ok_rows, key=ranking_key)
            for rank, row in enumerate(ranked, start=1):
                row["rank"] = rank
            payload["ranking"] = [
                {
                    "rank": row["rank"],
                    "model": row["model"],
                    "model_type": row.get("model_type"),
                    "hidden_size": row.get("hidden_size"),
                    "target_count": row.get("target_count"),
                    "fn_median": row.get("fn_median"),
                    "fn_p90": row.get("fn_p90"),
                    "fn_max": row.get("fn_max"),
                    "cap_scale_median": row.get("cap_scale_median"),
                    "cap_scale_p10": row.get("cap_scale_p10"),
                    "cap_scale_min": row.get("cap_scale_min"),
                    "le_cap_frac": row.get("le_cap_frac"),
                }
                for row in ranked
            ]
            write_json(args.output, payload)
            write_csv(csv_output, sorted(payload["results"], key=ranking_key))
            release(torch, model)

    print(f"wrote {args.output}", flush=True)
    print(f"wrote {csv_output}", flush=True)
    if payload["ranking"]:
        best = payload["ranking"][0]
        print(
            "best_by_p90={model} fn_p90={fn_p90:.3f} fn_max={fn_max:.3f} "
            "scale_median={cap_scale_median:.3f}".format(**best),
            flush=True,
        )


if __name__ == "__main__":
    main()
