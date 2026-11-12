"""An explicit complete Transformer layer for the draft's zero/one-mark family."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from quantum_friendly_transformer import UnitNormLayerNorm


def input_rows(n, marked_index=None):
    if n < 4 or (marked_index is not None and not 0 <= marked_index < n-1):
        raise ValueError('The final query is public and cannot be marked')
    rows = torch.zeros(n, 4, dtype=torch.float64)
    rows[:, 0] = 1 / math.sqrt(2)
    rows[:, 1] = -1 / math.sqrt(2)
    if marked_index is not None:
        rows[marked_index, 1] *= -1
    return rows


class SelectiveLayer(nn.Module):
    """Native d=4 attention, two residuals, two fixed unit-norm LNs, zero FFN.

    q and k have Frobenius norm sqrt(2 log N). The value projection has norm
    sqrt(2). The last query attends to all N rows, including its known -1 row.
    This is the selective regime, not the constant-logit denominator regime.
    """

    def __init__(self, n, eps=0.):
        super().__init__()
        self.n = n
        self.denominator = 2.
        for name in ['q', 'k', 'v', 'm1', 'm2']:
            module = nn.Linear(4, 4, bias=False, dtype=torch.float64)
            with torch.no_grad():
                module.weight.zero_()
            module.requires_grad_(False)
            setattr(self, name, module)
        with torch.no_grad():
            self.q.weight[0, 0] = math.sqrt(2 * math.log(n))
            self.k.weight[0, 1] = math.sqrt(2 * math.log(n))
            self.v.weight[2, 1] = math.sqrt(2)
        self.ln1 = UnitNormLayerNorm(4, eps=eps, dtype=torch.float64)
        self.ln2 = UnitNormLayerNorm(4, eps=eps, dtype=torch.float64)

    def finish(self, attended, query):
        first = query + attended
        hidden = self.ln1(first)
        second = hidden + self.m2(F.gelu(self.m1(hidden)))
        output = self.ln2(second)
        return output, {'sigma1': (first-first.mean(-1, keepdim=True)).norm(dim=-1),
                        'sigma2': (second-second.mean(-1, keepdim=True)).norm(dim=-1)}

    def forward(self, rows):
        assert rows.shape == (self.n, 4)
        query = rows[-1]
        logits = (self.q(query) @ self.k(rows).T) / self.denominator
        probabilities = logits.softmax(-1)
        attended = probabilities @ self.v(rows)
        output, conditioning = self.finish(attended, query)
        return output, {'logits': logits, 'probabilities': probabilities,
                        'attention_average': attended[2], **conditioning}

    def from_average(self, average):
        average = torch.as_tensor(average, dtype=torch.float64)
        attended = torch.zeros(*average.shape, 4, dtype=torch.float64)
        attended[..., 2] = average
        query = torch.tensor([1., -1., 0., 0.], dtype=torch.float64) / math.sqrt(2)
        return self.finish(attended, query)[0]


def grover_parameters(candidates):
    """Unique-mark rotation count; a final queried verification handles zero marks."""
    if candidates < 3:
        raise ValueError('Use at least three candidate rows')
    theta = math.asin(1 / math.sqrt(candidates))
    iterations = math.floor(math.pi / (4 * theta))
    probability = math.sin((2 * iterations + 1) * theta)**2
    return iterations, probability


def simulate_search_state(candidates, marked_index, iterations):
    """Dense classical state-vector simulation, used only at small sizes."""
    state = np.full(candidates, 1 / math.sqrt(candidates), dtype=np.float64)
    for _ in range(iterations):
        if marked_index is not None:
            state[marked_index] *= -1
        state = 2 * state.mean() - state
    return state

