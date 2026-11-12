"""Balanced long-context recall with exactly matched mean-pooling inputs."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from quantum_friendly_transformer import UnitNormLayerNorm, learnable_frobenius_cap


VARIANTS = ("strict_frobenius", "general_alpha1", "native_sqrt_d", "uniform")


def paired_batch(pairs, n, d, key_dimension, generator, device="cpu"):
    """Two sequences per fresh key set; reverse all balanced binary values.

    There are N-2 memory tokens, one public marker, and one query. Every memory
    token carries a random continuous key and a binary value. The two sequences
    have identical queries and identical row means, but opposite target labels.
    All N rows are visible, including the marker and query. No position encoding.
    Key norms have a small rounding margin; the value coordinate is exactly +/-
    1/2 so its balanced sum is exactly zero at the tested power-of-two lengths.
    """
    if n < 4 or n % 2 or d < key_dimension + 3:
        raise ValueError("Need even N >= 4 and room for value/query/marker coordinates")
    memory = n - 2
    keys = torch.randn(pairs, memory, key_dimension, generator=generator, device=device)
    keys = F.normalize(keys, dim=-1) * (math.sqrt(.75) * (1 - 2e-6))
    order = torch.rand(pairs, memory, generator=generator, device=device).argsort(-1)
    values = torch.where(order < memory // 2, -.5, .5)
    selected = torch.randint(memory, (pairs,), generator=generator, device=device)
    batch = torch.arange(pairs, device=device)
    base = torch.zeros(pairs, n, d, device=device)
    base[:, :memory, :key_dimension] = keys
    base[:, :memory, key_dimension] = values
    base[:, -2, key_dimension + 2] = 1 - 2e-6
    base[:, -1, :key_dimension] = keys[batch, selected]
    base[:, -1, key_dimension + 1] = .5
    rows = base.repeat_interleave(2, 0)
    rows[1::2, :memory, key_dimension] *= -1
    target = (values[batch, selected] > 0).long()
    labels = torch.stack((target, 1 - target), dim=1).flatten()
    return rows, labels, selected.repeat_interleave(2)


def input_certificate(rows):
    """Check every supplied row, including every training batch, in float64."""
    squared = rows.double().square().sum(-1)
    maximum_row_squared = float(squared.max())
    maximum_matrix_squared = float(squared.sum(-1).max())
    if maximum_row_squared > 1 or maximum_matrix_squared > rows.shape[1]:
        raise AssertionError("Certified unit-row / sqrt(N) Frobenius bound exceeded")
    return maximum_row_squared, maximum_matrix_squared


class LongRecallLayer(nn.Module):
    """Selected-row attention + residual/LN + GELU FFN + residual/LN + head.

    Q/K/V, both FFN matrices, and the downstream head are Frobenius capped.
    Both LayerNorms have fixed gamma=1/sqrt(d), beta=0, and epsilon=0.
    General-alpha1 is explicitly the general-scaling regime, not kappa=1.
    """

    def __init__(self, d=32, d_ff=64, cap=24., variant="strict_frobenius"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.d, self.cap, self.variant = d, cap, variant

        def linear(source, target):
            return learnable_frobenius_cap(nn.Linear(source, target, bias=False), max_gamma=cap)

        self.q = linear(d, d)
        self.k = linear(d, d)
        self.v = linear(d, d)
        self.m1 = linear(d, d_ff)
        self.m2 = linear(d_ff, d)
        self.classifier = linear(d, 2)
        self.ln1 = UnitNormLayerNorm(d, eps=0)
        self.ln2 = UnitNormLayerNorm(d, eps=0)

    def denominator(self, n):
        if self.variant == "strict_frobenius":
            return n * self.cap**2
        if self.variant == "native_sqrt_d":
            return math.sqrt(self.d)
        return 1.

    def finish(self, attended, query):
        first = query + attended
        hidden = self.ln1(first)
        second = hidden + self.m2(F.gelu(self.m1(hidden)))
        hidden = self.ln2(second)
        logits = self.classifier(hidden)
        return logits, hidden, {
            "sigma1": (first - first.mean(-1, keepdim=True)).norm(dim=-1),
            "sigma2": (second - second.mean(-1, keepdim=True)).norm(dim=-1),
        }

    def forward(self, rows, diagnostics=False, force_uniform=False):
        query = rows[:, -1]
        # Exact algebraic reassociation avoids materializing every projected K/V.
        # A separate dense implementation is checked in the validation suite.
        if self.variant == "uniform" or force_uniform:
            scores = rows.new_zeros(rows.shape[:2])
            probabilities = scores.new_full(scores.shape, 1 / rows.shape[1])
            pooled = rows.mean(1)
        else:
            coefficient = self.q(query) @ self.k.weight
            scores = (rows * coefficient[:, None]).sum(-1) / self.denominator(rows.shape[1])
            probabilities = scores.softmax(-1)
            pooled = (probabilities[:, None] @ rows).squeeze(1)
        attended = self.v(pooled)
        logits, hidden, conditioning = self.finish(attended, query)
        if not diagnostics:
            return logits
        return logits, {"scores": scores, "probabilities": probabilities,
                        "attended": attended, "hidden": hidden, **conditioning}

    def dense_forward(self, rows):
        query = rows[:, -1]
        if self.variant == "uniform":
            probabilities = rows.new_full(rows.shape[:2], 1 / rows.shape[1])
        else:
            scores = (self.q(query)[:, None] @ self.k(rows).transpose(-1, -2)).squeeze(1)
            probabilities = (scores / self.denominator(rows.shape[1])).softmax(-1)
        attended = (probabilities[:, None] @ self.v(rows)).squeeze(1)
        return self.finish(attended, query)


def matrix_records(model):
    records = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        raw = float(module.parametrizations.weight.original.detach().double().norm())
        effective = float(module.weight.detach().double().norm())
        gamma = float(module.parametrizations.weight[0].gamma.detach())
        if not math.isfinite(effective) or effective > model.cap * (1 + 1e-7):
            raise AssertionError(f"Matrix cap exceeded: {name}")
        records.append({"matrix": name, "raw_frobenius": raw,
                        "effective_frobenius": effective, "gamma": gamma,
                        "cap_active": raw > gamma})
    return records
