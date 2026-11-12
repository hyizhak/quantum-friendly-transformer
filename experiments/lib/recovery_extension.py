"""Data splits and diagnostics for the bounded recovery extension."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch


def sentence_key(text):
    return " ".join(text.split()).casefold()


def partition_training(records, evaluation_records, fraction, seed):
    """Keep duplicate sentences together and exclude exact evaluation overlaps."""
    if not 0 < fraction < 1:
        raise ValueError("selection fraction must be between zero and one")
    evaluation_keys = {sentence_key(r["sentence"]) for r in evaluation_records}
    groups = {}
    excluded = []
    for i, row in enumerate(records):
        key = sentence_key(row["sentence"])
        if key in evaluation_keys:
            excluded.append(i)
        else:
            groups.setdefault(key, []).append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    count = max(1, min(len(keys) - 1, round(len(keys) * fraction)))
    selected = set(keys[:count])
    train = sorted(i for key, indices in groups.items() if key not in selected for i in indices)
    selection = sorted(i for key, indices in groups.items() if key in selected for i in indices)
    if not train or not selection:
        raise ValueError("both training and selection sets must be nonempty")
    return {"train": train, "selection": selection, "excluded_evaluation_overlap": excluded}


def indices_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


class AttentionInputMonitor:
    """Measure whole-matrix and row norms during explicit diagnostic passes.

    The full padded matrix is measured. This supplies an upper bound for any
    visible-row subset but is not a certificate for unobserved training batches.
    No hooks are installed during the optimized training forwards.
    """

    def __init__(self, model, alpha_s):
        self.model = model
        self.alpha_s = alpha_s
        self.handles = []
        self.values = {}

    def __enter__(self):
        for name, module in self.model.named_modules():
            if type(module).__name__ == "GPTNeoSelfAttention":
                self.handles.append(module.register_forward_pre_hook(self._hook(name), with_kwargs=True))
        if not self.handles:
            raise ValueError("no GPT-Neo attention modules found")
        return self

    def _hook(self, name):
        def record(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None:
                raise ValueError("attention input is missing")
            x = hidden.detach().float()
            frob = torch.linalg.vector_norm(x, dim=(-2, -1))
            token = torch.linalg.vector_norm(x, dim=-1)
            values = torch.stack([frob.max(), token.max(), frob.min()])
            old = self.values.get(name)
            if old is not None:
                values = torch.stack([torch.maximum(values[0], old[0]),
                                      torch.maximum(values[1], old[1]),
                                      torch.minimum(values[2], old[2])])
            self.values[name] = values
        return record

    def rows(self):
        return [{"layer": name, "maximum_matrix_frobenius": float(v[0]),
                 "maximum_token_l2": float(v[1]), "minimum_matrix_frobenius": float(v[2]),
                 "alpha_s": self.alpha_s, "maximum_matrix_to_alpha_ratio": float(v[0]) / self.alpha_s}
                for name, v in self.values.items()]

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()


def install_uniform_attention(model):
    count = 0
    for module in model.modules():
        if type(module).__name__ == "GPTNeoSelfAttention":
            module.q_proj.register_forward_hook(lambda _module, _args, output: output * 0)
            count += 1
    if not count:
        raise ValueError("no GPT-Neo attention modules found")
    return count


def validate_scale_records(records, alpha_s, cap):
    floor = alpha_s**2 * cap**2
    if not records:
        raise ValueError("bounded run has no scale controllers")
    for row in records:
        if not row["alpha_denominator_bound_enforced"]:
            raise ValueError("unbounded attention controller")
        if not math.isfinite(row["score_scale"]) or row["score_scale"] > 1 / floor:
            raise ValueError("score scale escaped the configured alpha floor")
        if row["score_denominator"] < floor:
            raise ValueError("denominator is below the configured alpha floor")
