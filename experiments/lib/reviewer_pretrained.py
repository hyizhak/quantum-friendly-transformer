"""Instrumentation for GPT-Neo; records actual masks and pre-LayerNorm tensors."""

from __future__ import annotations

import math
import types

import numpy as np
import torch

from experiments.lib.reviewer import attention_statistics


def quantiles(tensor):
    x = tensor.detach().float().reshape(-1)
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite diagnostic measurement")
    values = torch.quantile(x, x.new_tensor([0, 0.01, 0.05, 0.5, 0.95, 1])).tolist()
    return dict(zip(["min", "p01", "p05", "median", "p95", "max"], values))


def perturb_interface(hidden, epsilon_rel, kind, generator=None):
    """Bound perturbations after expressing each residual row in unit-norm coordinates.

    For x != 0, write u=x/||x||. The injected coordinate error in u is at most
    epsilon_rel/sqrt(d), and the output is ||x||*(u+error). No renormalization.
    Uniform noise is a sensitivity surrogate, not a tomography simulation.
    """
    if epsilon_rel < 0:
        raise ValueError("epsilon_rel must be nonnegative")
    if epsilon_rel == 0:
        return hidden
    delta = epsilon_rel * hidden.float().square().mean(dim=-1, keepdim=True).sqrt()
    if kind == "uniform":
        noise = torch.rand(hidden.shape, generator=generator, device=hidden.device, dtype=torch.float32) * 2 - 1
        return hidden + (delta * noise).to(hidden.dtype)
    if kind == "rounding":
        step = 2 * delta
        safe_step = step.clamp_min(torch.finfo(step.dtype).tiny)
        rounded = torch.round(hidden.float() / safe_step) * step
        return torch.where(step > 0, rounded.to(hidden.dtype), hidden)
    raise ValueError(f"Unknown perturbation kind: {kind}")


class InterfaceNoise:
    def __init__(self, model, depth, epsilon_rel, kind, seed):
        self.handles = []
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.max_bound_ratio = 0.0

        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            changed = perturb_interface(hidden, epsilon_rel, kind, self.generator)
            scale = epsilon_rel * hidden.float().square().mean(dim=-1, keepdim=True).sqrt()
            ratio = (changed - hidden).abs() / scale.clamp_min(1e-20)
            self.max_bound_ratio = max(self.max_bound_ratio, float(ratio.max()))
            if self.max_bound_ratio > 1.002:
                raise AssertionError("Injected noise exceeded its configured bound")
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        if not 0 <= depth <= len(model.transformer.h):
            raise ValueError("Noise depth exceeds the model's layers")
        for block in model.transformer.h[:depth]:
            self.handles.append(block.register_forward_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()


class GPTNeoDiagnostics:
    def __init__(self, model, variant, sequence_length, alpha_s_bound=None):
        self.context = {"variant": variant, "N": sequence_length, "block": None}
        self.attention, self.normalization, self.residual, self.inputs = [], [], [], []
        self.handles, self.original_attn = [], []
        self.alpha_s_bound = alpha_s_bound
        self.attention_types = list(model.config.attention_layers)
        for layer, block in enumerate(model.transformer.h):
            self.handles.append(block.register_forward_pre_hook(self._residual_hook(layer, "block_input")))
            self.handles.append(block.register_forward_hook(self._residual_hook(layer, "block_output")))
            for norm_name in ["ln_1", "ln_2"]:
                self.handles.append(getattr(block, norm_name).register_forward_pre_hook(self._norm_hook(layer, norm_name)))
            attn = block.attn.attention
            self.handles.append(attn.q_proj.register_forward_pre_hook(self._input_hook(layer)))
            original = attn._attn
            self.original_attn.append((attn, original))
            attn._attn = types.MethodType(self._attention_wrapper(layer, original), attn)

    def _residual_hook(self, layer, boundary):
        def hook(module, args, output=None):
            x = args[0] if boundary == "block_input" else output[0] if isinstance(output, tuple) else output
            self.residual.append({**self.context, "layer": layer, "boundary": boundary,
                                  **quantiles(x.norm(dim=-1)), "matrix_frobenius_norm": float(x.norm()),
                                  "max_absolute_activation": float(x.abs().max())})
        return hook

    def _norm_hook(self, layer, name):
        def hook(module, args):
            x = args[0].float()
            centered = x - x.mean(dim=-1, keepdim=True)
            sigma = centered.norm(dim=-1)
            denom = (centered.square().mean(dim=-1) + module.eps).sqrt()
            self.normalization.append({**self.context, "layer": layer, "normalization": name,
                                       "epsilon_ln": module.eps,
                                       **{f"centered_norm_{k}": v for k, v in quantiles(sigma).items()},
                                       **{f"native_denominator_{k}": v for k, v in quantiles(denom).items()}})
        return hook

    def _input_hook(self, layer):
        def hook(module, args):
            x = args[0]
            fro = float(x.float().norm())
            if self.alpha_s_bound is not None and fro > self.alpha_s_bound * (1 + 1e-5):
                raise AssertionError("Configured alpha_s is smaller than the measured sequence norm")
            self.inputs.append({**self.context, "layer": layer, "frobenius_norm": fro,
                                "max_token_norm": float(x.float().norm(dim=-1).max()),
                                "alpha_s_bound": self.alpha_s_bound if self.alpha_s_bound is not None else "",
                                "bound_ratio": fro / self.alpha_s_bound if self.alpha_s_bound is not None else ""})
        return hook

    def _attention_wrapper(self, layer, original):
        def wrapper(module, query, key, value, attention_mask=None, head_mask=None):
            # Inspect only the final query to avoid storing an N x N diagnostic tensor.
            # Query already includes the library's installed scaling hook.
            logits = torch.matmul(query[:, :, -1:, :].float(), key.transpose(-1, -2).float())[0, :, 0]
            visible = module.bias[0, 0, key.shape[-2] - 1, :key.shape[-2]].bool()
            if attention_mask is not None:
                mask = attention_mask[0, 0, -1, :key.shape[-2]]
                visible = visible & (mask > -1e4)
                logits = logits + mask
            for head in range(logits.shape[0]):
                self.attention.append({**self.context, "layer": layer, "head": head,
                                       "query_position": key.shape[-2] - 1,
                                       "attention_type": self.attention_types[layer],
                                       **attention_statistics(logits[head, visible].detach().double().cpu().numpy())})
            return original(query, key, value, attention_mask, head_mask)
        return wrapper

    def close(self):
        for handle in self.handles:
            handle.remove()
        for module, original in self.original_attn:
            module._attn = original
