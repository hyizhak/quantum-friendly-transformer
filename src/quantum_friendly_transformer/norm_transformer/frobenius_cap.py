from __future__ import annotations

from collections.abc import Callable, Iterable
import math

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize


TargetFilter = Callable[[str, nn.Module], bool]


def _logit_for_gamma(init_gamma: float, max_gamma: float) -> Tensor:
    ratio = min(max(init_gamma / max_gamma, 1e-6), 1.0 - 1e-6)
    return torch.logit(torch.tensor(ratio, dtype=torch.float32))


class LearnableFrobeniusCap(nn.Module):
    """Parametrization that caps a weight's Frobenius norm with a learnable gamma.

    The exposed weight is `min(1, gamma / ||W||_F) * W`, where
    `0 < gamma < max_gamma`. This differs from full Frobenius normalization:
    weights already below the cap pass through unchanged.
    """

    def __init__(
        self,
        max_gamma: float,
        init_fraction: float = 16.0 / 17.0,
        eps: float = 1e-12,
        norm_dtype: str = "float32",
    ) -> None:
        super().__init__()
        if not math.isfinite(max_gamma) or max_gamma <= 0:
            raise ValueError("max_gamma must be positive and finite")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be positive and finite")
        if not math.isfinite(init_fraction) or not 0 < init_fraction < 1:
            raise ValueError("init_fraction must be between 0 and 1")
        if norm_dtype not in {"float32", "native"}:
            raise ValueError("norm_dtype must be 'float32' or 'native'")
        self.max_gamma = float(max_gamma)
        self.eps = eps
        self.norm_dtype = norm_dtype
        init_gamma = max(1e-6, self.max_gamma * init_fraction)
        self.g = nn.Parameter(_logit_for_gamma(init_gamma, self.max_gamma))

    @property
    def gamma(self) -> Tensor:
        return self.max_gamma * torch.sigmoid(self.g)

    def forward(self, weight: Tensor) -> Tensor:
        gamma = self.gamma
        norm_weight = weight.float() if self.norm_dtype == "float32" and weight.dtype != torch.float64 else weight
        norm = torch.linalg.vector_norm(norm_weight)
        if norm.dtype in (torch.float16, torch.bfloat16):
            # Keep the requested reduction dtype, but eps/division must not
            # underflow at zero (otherwise the backward pass produces NaNs).
            norm = norm.float()
        scale = torch.clamp(gamma / (norm + self.eps), max=1.0)
        return scale.to(weight.dtype) * weight


class OutputFrobeniusCapWrapper(nn.Module):
    """Output-scaling fallback for quantized linear modules.

    Quantized modules often cannot be parametrized safely. This wrapper keeps
    the module intact and scales its output using a learnable cap derived from
    the dequantized base Frobenius norm.
    """

    def __init__(
        self,
        module: nn.Module,
        max_gamma: float,
        base_frobenius_norm: float,
        init_fraction: float = 16.0 / 17.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if max_gamma <= 0:
            raise ValueError("max_gamma must be positive")
        self.module = module
        self.max_gamma = float(max_gamma)
        self.eps = eps
        self.register_buffer(
            "base_frobenius_norm",
            torch.tensor(float(base_frobenius_norm), dtype=torch.float32),
            persistent=False,
        )
        init_gamma = max(1e-6, self.max_gamma * init_fraction)
        self.g = nn.Parameter(_logit_for_gamma(init_gamma, self.max_gamma))

    @property
    def gamma(self) -> Tensor:
        return self.max_gamma * torch.sigmoid(self.g)

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        gamma = self.gamma
        base_norm = self.base_frobenius_norm.to(device=gamma.device)
        scale = torch.clamp(gamma / (base_norm + self.eps), max=1.0)
        return output * scale.to(dtype=output.dtype, device=output.device)


def _is_adapter_internal_name(lowered: str) -> bool:
    return ".base_layer" in lowered or ".lora_" in lowered


def is_transformer_target_module_name(module_name: str) -> bool:
    """Return True for standard attention/MLP projection modules."""

    lowered = module_name.lower()
    encoder_target = (
        (".encoder.layer." in lowered or lowered.startswith("encoder.layer."))
        and (
            lowered.endswith("attention.self.query")
            or lowered.endswith("attention.self.key")
            or lowered.endswith("attention.self.value")
            or lowered.endswith("attention.output.dense")
            or lowered.endswith("intermediate.dense")
            or lowered.endswith("output.dense")
        )
    )
    gpt2_target = (
        (".h." in lowered or lowered.startswith("h."))
        and (
            lowered.endswith(".attn.c_attn")
            or lowered.endswith(".attn.c_proj")
            or lowered.endswith(".mlp.c_fc")
            or lowered.endswith(".mlp.c_proj")
        )
    )
    llama_target = (
        (".layers." in lowered or lowered.startswith("layers."))
        and (
            lowered.endswith(".self_attn.q_proj")
            or lowered.endswith(".self_attn.k_proj")
            or lowered.endswith(".self_attn.v_proj")
            or lowered.endswith(".self_attn.o_proj")
            or lowered.endswith(".mlp.gate_proj")
            or lowered.endswith(".mlp.up_proj")
            or lowered.endswith(".mlp.down_proj")
        )
    )
    mixtral_target = (
        (".layers." in lowered or lowered.startswith("layers."))
        and (
            lowered.endswith(".self_attn.q_proj")
            or lowered.endswith(".self_attn.k_proj")
            or lowered.endswith(".self_attn.v_proj")
            or lowered.endswith(".self_attn.o_proj")
            or (
                ".block_sparse_moe.experts." in lowered
                and (lowered.endswith(".w1") or lowered.endswith(".w2") or lowered.endswith(".w3"))
            )
        )
    )
    opt_target = (
        (".decoder.layers." in lowered or lowered.startswith("decoder.layers."))
        and (
            lowered.endswith(".self_attn.q_proj")
            or lowered.endswith(".self_attn.k_proj")
            or lowered.endswith(".self_attn.v_proj")
            or lowered.endswith(".self_attn.out_proj")
            or lowered.endswith(".fc1")
            or lowered.endswith(".fc2")
        )
    )
    gpt_neo_target = (
        (".h." in lowered or lowered.startswith("h."))
        and (
            lowered.endswith(".attn.attention.q_proj")
            or lowered.endswith(".attn.attention.k_proj")
            or lowered.endswith(".attn.attention.v_proj")
            or lowered.endswith(".attn.attention.out_proj")
            or lowered.endswith(".mlp.c_fc")
            or lowered.endswith(".mlp.c_proj")
        )
    )
    gpt_neox_target = (
        (".layers." in lowered or lowered.startswith("layers."))
        and (
            lowered.endswith(".attention.query_key_value")
            or lowered.endswith(".attention.dense")
            or lowered.endswith(".mlp.dense_h_to_4h")
            or lowered.endswith(".mlp.dense_4h_to_h")
        )
    )
    bloom_target = (
        (".h." in lowered or lowered.startswith("h."))
        and (
            ".self_attention.query_key_value" in lowered
            or ".self_attention.dense" in lowered
            or ".mlp.dense_h_to_4h" in lowered
            or ".mlp.dense_4h_to_h" in lowered
        )
    )
    phi_target = (
        (".layers." in lowered or lowered.startswith("layers."))
        and (
            ".self_attn.q_proj" in lowered
            or ".self_attn.k_proj" in lowered
            or ".self_attn.v_proj" in lowered
            or ".self_attn.dense" in lowered
            or ".mlp.fc1" in lowered
            or ".mlp.fc2" in lowered
        )
    )
    xglm_target = (".layers." in lowered or lowered.startswith("layers.")) and (
        ".self_attn.out_proj" in lowered or ".fc1" in lowered or ".fc2" in lowered
    )
    return (
        encoder_target
        or gpt2_target
        or llama_target
        or mixtral_target
        or opt_target
        or gpt_neo_target
        or gpt_neox_target
        or bloom_target
        or phi_target
        or xglm_target
    )


def is_transformer_target_parameter_name(parameter_name: str) -> bool:
    lowered = parameter_name.lower()
    if not lowered.endswith(".weight"):
        return False
    return is_transformer_target_module_name(lowered[: -len(".weight")])


def _default_target_filter(module_name: str, module: nn.Module) -> bool:
    return is_transformer_target_module_name(module_name)


def target_named_modules(model: nn.Module, target_filter: TargetFilter | None = None) -> Iterable[tuple[str, nn.Module]]:
    target_filter = target_filter or _default_target_filter
    for module_name, module in model.named_modules():
        lowered = module_name.lower()
        if _is_adapter_internal_name(lowered):
            continue
        if target_filter(module_name, module):
            yield module_name, module


def target_weight_modules(model: nn.Module, target_filter: TargetFilter | None = None) -> Iterable[tuple[str, nn.Module]]:
    for module_name, module in target_named_modules(model, target_filter=target_filter):
        weight = getattr(module, "weight", None)
        if weight is not None and getattr(weight, "ndim", None) == 2:
            yield module_name, module


def target_weight_parameters(model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    for name, param in model.named_parameters():
        lowered = name.lower()
        if _is_adapter_internal_name(lowered):
            continue
        if param.ndim == 2 and is_transformer_target_parameter_name(name):
            yield name, param


def _module_class_path(module: nn.Module) -> str:
    cls = type(module)
    return f"{cls.__module__}.{cls.__name__}".lower()


def _base_linear_module(module: nn.Module) -> nn.Module:
    return getattr(module, "base_layer", module)


def _is_quantized_linear_module(module: nn.Module) -> bool:
    base = _base_linear_module(module)
    class_path = _module_class_path(base)
    weight = getattr(base, "weight", None)
    return (
        "bitsandbytes" in class_path
        or "awq" in class_path
        or hasattr(base, "qweight")
        or hasattr(weight, "quant_state")
    )


def _dequantized_weight_for_norm(module: nn.Module) -> Tensor | None:
    base = _base_linear_module(module)
    if hasattr(base, "qweight") and hasattr(base, "qzeros") and hasattr(base, "scales"):
        from awq.utils.packing_utils import dequantize_gemm

        return dequantize_gemm(base.qweight, base.qzeros, base.scales, base.w_bit, base.group_size)

    weight = getattr(base, "weight", None)
    if weight is None or getattr(weight, "ndim", None) != 2:
        return None
    if hasattr(weight, "dequantize"):
        return weight.dequantize()
    if hasattr(weight, "quant_state"):
        import bitsandbytes.functional as bnb_functional

        return bnb_functional.dequantize_4bit(weight.data, weight.quant_state)
    return weight


def estimate_module_frobenius_norm(module: nn.Module) -> float | None:
    with torch.no_grad():
        weight = _dequantized_weight_for_norm(module)
        if weight is None:
            return None
        value = float(torch.linalg.vector_norm(weight.float()).detach().cpu().item())
        del weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return value


def _set_module_by_name(model: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent._modules[part]
    parent._modules[parts[-1]] = replacement


def learnable_frobenius_cap(
    module: nn.Module,
    max_gamma: float,
    name: str = "weight",
    init_fraction: float = 16.0 / 17.0,
    eps: float = 1e-12,
    norm_dtype: str = "float32",
) -> nn.Module:
    weight = getattr(module, name, None)
    if weight is None or getattr(weight, "ndim", None) != 2:
        raise ValueError(f"module has no 2D parameter or buffer named {name!r}")
    if parametrize.is_parametrized(module, name):
        return module
    parametrize.register_parametrization(
        module,
        name,
        LearnableFrobeniusCap(
            max_gamma=max_gamma,
            init_fraction=init_fraction,
            eps=eps,
            norm_dtype=norm_dtype,
        ),
    )
    return module


def apply_learnable_frobenius_caps(
    model: nn.Module,
    max_gamma: float,
    target_filter: TargetFilter | None = None,
    init_fraction: float = 16.0 / 17.0,
    eps: float = 1e-12,
    norm_dtype: str = "float32",
    allow_output_wrapper: bool = True,
) -> int:
    """Apply learnable Frobenius caps to transformer attention/MLP weights."""

    applied = 0
    for module_name, module in list(target_named_modules(model, target_filter=target_filter)):
        if isinstance(module, OutputFrobeniusCapWrapper):
            continue
        weight = getattr(module, "weight", None)
        can_parametrize = (
            weight is not None
            and getattr(weight, "ndim", None) == 2
            and not _is_quantized_linear_module(module)
            and not hasattr(module, "base_layer")
        )
        if can_parametrize:
            if parametrize.is_parametrized(module, "weight"):
                continue
            learnable_frobenius_cap(
                module,
                max_gamma=max_gamma,
                init_fraction=init_fraction,
                eps=eps,
                norm_dtype=norm_dtype,
            )
            applied += 1
            continue

        if not allow_output_wrapper:
            continue
        base_norm = estimate_module_frobenius_norm(module)
        if base_norm is None:
            continue
        _set_module_by_name(
            model,
            module_name,
            OutputFrobeniusCapWrapper(
                module,
                max_gamma=max_gamma,
                base_frobenius_norm=base_norm,
                init_fraction=init_fraction,
                eps=eps,
            ),
        )
        applied += 1
    return applied


def collect_learnable_cap_stats(model: nn.Module) -> list[dict]:
    rows = []
    for name, module in target_weight_modules(model):
        if not parametrize.is_parametrized(module, "weight"):
            continue
        for index, parametrization in enumerate(module.parametrizations.weight):
            if not hasattr(parametrization, "g") or not hasattr(parametrization, "max_gamma"):
                continue
            gamma = float(parametrization.gamma.detach().float().cpu().item())
            rows.append(
                {
                    "name": f"{name}.weight",
                    "parametrization_index": index,
                    "max_gamma": float(parametrization.max_gamma),
                    "gamma": gamma,
                    "gamma_fraction": gamma / float(parametrization.max_gamma),
                    "cap_type": "parametrization",
                }
            )
    for name, module in model.named_modules():
        if not isinstance(module, OutputFrobeniusCapWrapper):
            continue
        gamma = float(module.gamma.detach().float().cpu().item())
        base_norm = float(module.base_frobenius_norm.detach().cpu().item())
        rows.append(
            {
                "name": f"{name}.output",
                "parametrization_index": 0,
                "max_gamma": float(module.max_gamma),
                "gamma": gamma,
                "gamma_fraction": gamma / float(module.max_gamma),
                "base_frobenius_norm": base_norm,
                "scale": min(gamma / max(base_norm, 1e-12), 1.0),
                "cap_type": "output_scale",
            }
        )
    return rows


def _weight_norms(param: Tensor) -> tuple[float, float, float]:
    weight = param.detach().float()
    fro = torch.linalg.vector_norm(weight).item()
    try:
        spectral = torch.linalg.matrix_norm(weight, ord=2).item()
    except RuntimeError:
        spectral = float("nan")
    col = torch.linalg.vector_norm(weight, dim=0).max().item()
    return fro, spectral, col


def collect_weight_stats(model: nn.Module, target_filter: TargetFilter | None = None) -> list[dict]:
    rows = []
    for name, module in target_weight_modules(model, target_filter=target_filter):
        weight = getattr(module, "weight")
        fro, spectral, col = _weight_norms(weight)
        rows.append(
            {
                "name": f"{name}.weight",
                "frobenius_norm": fro,
                "spectral_norm": spectral,
                "max_col_norm": col,
                "shape": list(weight.shape),
            }
        )
    return rows
