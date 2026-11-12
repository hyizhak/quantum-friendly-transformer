"""Numerical and artifact helpers for reproducible reviewer experiments."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def attention_statistics(logits):
    logits = np.asarray(logits, dtype=np.float64)
    p = softmax(logits)
    entropy = float(-np.sum(p * np.log(np.maximum(p, np.finfo(float).tiny))))
    return {"visible_keys": len(p), "logit_min": float(logits.min()),
            "logit_max": float(logits.max()), "logit_range": float(np.ptp(logits)),
            "max_probability": float(p.max()), "max_probability_times_keys": float(p.max() * len(p)),
            "entropy": entropy, "normalized_entropy": entropy / math.log(len(p)) if len(p) > 1 else 1.0,
            "participation_ratio": float(1.0 / np.sum(p * p)),
            "log_partition": float(logits.max() + np.log(np.exp(logits - logits.max()).sum()))}


def sampled_attention(rows, query_coefficient, value_weight, indices):
    """Uniform row sampler; no access to precomputed logits or value rows."""
    sampled_rows = rows[indices]
    probabilities = softmax(sampled_rows @ query_coefficient)
    return (probabilities @ sampled_rows) @ value_weight


def taylor_degree(kappa, delta):
    """Sufficient degree for exp(kappa*x/2), x in [-1,1], via Taylor remainder."""
    if kappa < 0 or not 0 < delta < 1:
        raise ValueError("Require kappa >= 0 and 0 < delta < 1")
    if kappa == 0:
        return 0
    a = kappa / 2

    def sufficient(degree):
        return a + (degree + 1) * math.log(a) - math.lgamma(degree + 2) <= math.log(delta)

    low, high = max(0, int(a)), max(1, int(4 * a + 10))
    while not sufficient(high):
        high *= 2
    while low < high:
        mid = (low + high) // 2
        if sufficient(mid):
            high = mid
        else:
            low = mid + 1
    return low


def save_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def manifest(config, source_files, **extra):
    versions = {}
    for package in ["numpy", "torch", "transformers", "matplotlib", "scipy", "huggingface-hub"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"utc": datetime.now(timezone.utc).isoformat(), "config": config,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files},
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": versions, **extra}
