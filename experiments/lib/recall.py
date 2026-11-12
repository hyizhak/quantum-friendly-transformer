"""A controlled associative-recall task with unit rows and bounded projections."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from quantum_friendly_transformer import UnitNormLayerNorm, learnable_frobenius_cap


def recall_batch(batch_size, n, d, key_dimension, classes, generator):
    """n-1 key/value rows followed by one query; predict the matched key's value.

    Keys are fresh continuous random vectors, so evaluation keys cannot be
    memorized. Each input row has Euclidean norm one. Query and key/value rows
    share their key coordinates, with the remaining norm allocated to a value
    one-hot vector or a query-type indicator. The query itself is not a key.
    """
    if n < 2 or d < key_dimension + classes + 1:
        raise ValueError("Invalid recall dimensions")
    keys = torch.randn(batch_size, n - 1, key_dimension, generator=generator)
    keys = F.normalize(keys, dim=-1)
    values = torch.randint(classes, (batch_size, n - 1), generator=generator)
    selected = torch.randint(n - 1, (batch_size,), generator=generator)
    rows = torch.zeros(batch_size, n, d)
    rows[:, :-1, :key_dimension] = keys / math.sqrt(2)
    rows[:, :-1, key_dimension:key_dimension + classes] = F.one_hot(values, classes) / math.sqrt(2)
    batch = torch.arange(batch_size)
    rows[:, -1, :key_dimension] = keys[batch, selected] / math.sqrt(2)
    rows[:, -1, key_dimension + classes] = 1 / math.sqrt(2)
    return rows, values[batch, selected], selected


class RecallLayer(nn.Module):
    """One selected-row attention/residual/LN/FFN/residual/LN block plus a head.

    All matrices including the task head have a Frobenius cap. There are no
    biases, positional terms, attention-output projection, or dropout. The head
    is a downstream classifier, not part of the analyzed attention routine.
    """

    def __init__(self, d, d_ff, classes, cap, scaling):
        super().__init__()
        if scaling not in {"native_sqrt_head_dim", "theorem_frobenius", "uniform"}:
            raise ValueError("Unknown scaling")
        self.d, self.cap, self.scaling = d, cap, scaling

        def linear(in_dim, out_dim):
            return learnable_frobenius_cap(nn.Linear(in_dim, out_dim, bias=False), max_gamma=cap)

        self.q = linear(d, d)
        self.k = linear(d, d)
        self.v = linear(d, d)
        self.m1 = linear(d, d_ff)
        self.m2 = linear(d_ff, d)
        self.ln1 = UnitNormLayerNorm(d)
        self.ln2 = UnitNormLayerNorm(d)
        self.classifier = linear(d, classes)

    def forward(self, rows, diagnostics=False):
        q = self.q(rows[:, -1:])
        k = self.k(rows[:, :-1])
        if self.scaling == "native_sqrt_head_dim":
            logits = (q @ k.transpose(-1, -2)).squeeze(1) / math.sqrt(self.d)
        elif self.scaling == "theorem_frobenius":
            # alpha_s=sqrt(N) is certified by the task's unit-norm rows;
            # alpha_w=cap is a bound on every complete Q/K/V matrix.
            logits = (q @ k.transpose(-1, -2)).squeeze(1) / (rows.shape[1] * self.cap**2)
        else:
            logits = rows.new_zeros((rows.shape[0], rows.shape[1] - 1))
        p = logits.softmax(dim=-1)
        attended = (p.unsqueeze(1) @ self.v(rows[:, :-1])).squeeze(1)
        first = rows[:, -1] + attended
        hidden = self.ln1(first)
        second = hidden + self.m2(F.gelu(self.m1(hidden)))
        hidden = self.ln2(second)
        output = self.classifier(hidden)
        if not diagnostics:
            return output
        return output, {"probabilities": p, "logits": logits,
                        "sigma1": (first - first.mean(-1, keepdim=True)).norm(dim=-1),
                        "sigma2": (second - second.mean(-1, keepdim=True)).norm(dim=-1)}


def matrix_stats(model):
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        effective = float(module.weight.detach().norm())
        raw = float(module.parametrizations.weight.original.detach().norm())
        gamma = float(module.parametrizations.weight[0].gamma.detach())
        if effective > model.cap * (1 + 1e-5):
            raise AssertionError("A trained matrix exceeded its configured cap")
        rows.append({"matrix": name, "raw_frobenius_norm": raw,
                     "effective_frobenius_norm": effective, "gamma": gamma,
                     "cap_active": int(raw > gamma)})
    return rows
