"""Official MQAR data adapter and capped causal Transformer under evaluation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from quantum_friendly_transformer import UnitNormLayerNorm, learnable_frobenius_cap

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "experiments/vendor/zoology"


@dataclass
class DataSegment:
    inputs: torch.Tensor
    labels: torch.Tensor
    slices: dict


def official_generator():
    """Load the unchanged upstream function, without the Zoology training stack."""
    path = VENDOR / "multiquery_ar.py"
    data = path.read_bytes()
    expected = json.loads((VENDOR / "UPSTREAM.json").read_text())["files"][path.name]["git_blob_sha"]
    actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if actual != expected:
        raise ValueError("Vendored MQAR source differs from its pinned Git blob")
    parsed = ast.parse(data, filename=str(path))
    function = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "multiquery_ar")
    scope = {"np": np, "torch": torch, "DataSegment": DataSegment}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope["multiquery_ar"]


def generate_data(config, *, n, pairs, examples, seed):
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            # manual_seed would also alter CUDA RNGs, which are not forked here.
            torch.random.default_generator.manual_seed(seed)
            return official_generator()(
                vocab_size=config["vocab_size"], num_examples=examples,
                input_seq_len=n, seed=seed, num_kv_pairs=pairs,
                power_a=config["power_a"], random_non_queries=config["random_non_queries"],
                num_passes=1,
            )
    finally:
        np.random.set_state(numpy_state)


def validate_data(segment, pairs, vocab_size):
    x, y = segment.inputs, segment.labels
    if x.shape != y.shape or not torch.all((y != -100).sum(-1) == pairs):
        raise AssertionError("MQAR query count/shape mismatch")
    if not bool(((x >= 0) & (x < vocab_size)).all()):
        raise AssertionError("Input outside the benchmark vocabulary")
    for inputs, labels in zip(x, y):
        keys, values = inputs[:2*pairs:2], inputs[1:2*pairs:2]
        mapping = dict(zip(keys.tolist(), values.tolist()))
        if len(mapping) != pairs or len(set(values.tolist())) != pairs:
            raise AssertionError("MQAR memory keys/values are not unique")
        positions = (labels != -100).nonzero().flatten()
        if int(positions.min()) < 2 * pairs:
            raise AssertionError("A scored query precedes its memory")
        if any(mapping.get(int(inputs[p])) != int(labels[p]) for p in positions):
            raise AssertionError("Official query label does not match its memory")


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class CappedBlock(nn.Module):
    def __init__(self, d, d_ff, cap):
        super().__init__()
        self.d = d
        def linear(a, b):
            return learnable_frobenius_cap(nn.Linear(a, b, bias=False), max_gamma=cap)
        self.q, self.k, self.v, self.o = [linear(d, d) for _ in range(4)]
        self.m1, self.m2 = linear(d, d_ff), linear(d_ff, d)
        self.ln1, self.ln2 = UnitNormLayerNorm(d, eps=0), UnitNormLayerNorm(d, eps=0)

    def forward(self, x, *, uniform=False, dense=False, diagnostics=False):
        v = self.v(x)
        if uniform:
            divisor = torch.arange(1, x.shape[1] + 1, dtype=x.dtype, device=x.device)[None, :, None]
            attended = v.cumsum(1) / divisor
        else:
            q, k = self.q(x), self.k(x)
            if dense:
                score = q @ k.transpose(-1, -2) / math.sqrt(self.d)
                mask = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
                attended = score.masked_fill(mask, -torch.inf).softmax(-1) @ v
            else:
                attended = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None], is_causal=True).squeeze(1)
        first = x + self.o(attended)
        hidden = self.ln1(first)
        second = hidden + self.m2(F.gelu(self.m1(hidden)))
        output = self.ln2(second)
        if diagnostics:
            return output, {
                "sigma1_min": float((first - first.mean(-1, keepdim=True)).norm(dim=-1).min()),
                "sigma2_min": float((second - second.mean(-1, keepdim=True)).norm(dim=-1).min()),
                "input_row_norm_max": float(x.double().norm(dim=-1).max()),
                "output_row_norm_max": float(output.double().norm(dim=-1).max()),
            }
        return output


class MQARTransformer(nn.Module):
    def __init__(self, config, variant="native"):
        super().__init__()
        if variant not in ("native", "uniform"):
            raise ValueError(variant)
        self.config, self.variant = config, variant
        self.embedding = nn.Embedding(config["vocab_size"], config["dimension"])
        self.blocks = nn.ModuleList([
            CappedBlock(config["dimension"], config["ffn_dimension"], config["cap"])
            for _ in range(config["layers"])
        ])
        self.readout = nn.Linear(config["dimension"], config["vocab_size"], bias=False)

    def embed(self, tokens):
        d = self.config["dimension"]
        position = torch.arange(tokens.shape[1], device=tokens.device, dtype=self.embedding.weight.dtype)
        frequency = torch.exp(torch.arange(0, d, 2, device=tokens.device, dtype=position.dtype) * (-math.log(10000.) / d))
        angle = position[:, None] * frequency[None]
        sinusoidal = torch.stack((angle.sin(), angle.cos()), -1).flatten(-2) / math.sqrt(d / 2)
        token = F.normalize(self.embedding(tokens), dim=-1)
        return F.normalize(token + sinusoidal, dim=-1) * (1 - 2e-6)

    def forward(self, tokens, positions=None, *, force_uniform=False, dense=False, diagnostics=False):
        hidden = self.embed(tokens)
        stats = []
        for block in self.blocks:
            result = block(hidden, uniform=force_uniform or self.variant == "uniform", dense=dense, diagnostics=diagnostics)
            if diagnostics:
                hidden, info = result
                stats.append(info)
            else:
                hidden = result
        selected = hidden if positions is None else hidden[positions]
        logits = self.readout(selected)
        return (logits, stats) if diagnostics else logits


def weight_records(model):
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        capped = hasattr(module, "parametrizations")
        raw_weight = module.parametrizations.weight.original if capped else module.weight
        raw = float(raw_weight.detach().double().norm())
        effective = float(module.weight.detach().double().norm())
        gamma = float(module.parametrizations.weight[0].gamma.detach()) if capped else None
        if capped and effective > model.config["cap"] * (1 + 1e-6):
            raise AssertionError(f"Cap exceeded by {name}")
        rows.append({"matrix": name, "capped": capped, "raw_frobenius": raw,
                     "effective_frobenius": effective, "gamma": gamma,
                     "cap_active": raw > gamma if capped else False})
    rows.append({"matrix": "embedding", "capped": False,
                 "raw_frobenius": float(model.embedding.weight.detach().double().norm()),
                 "effective_frobenius": math.sqrt(model.config["vocab_size"]),
                 "gamma": None, "cap_active": False})
    return rows
