"""Normalization layers with the quantum-friendly unit-norm affine constraint.

For a normalization width ``d``, the constrained affine parameters are
``gamma_i = 1 / sqrt(d)`` and ``beta_i = 0``. The affine values are buffers
rather than parameters, so optimizers and checkpoint loading cannot move them
away from the constraint.
"""

import math
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


NormalizedShape = Union[int, Sequence[int], torch.Size]


def _canonical_shape(normalized_shape: NormalizedShape) -> Tuple[int, ...]:
    if isinstance(normalized_shape, int):
        shape = (normalized_shape,)
    else:
        shape = tuple(int(size) for size in normalized_shape)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(
            f"normalized_shape must contain positive dimensions, got {shape}"
        )
    return shape


def _affine_scale(normalized_shape: Tuple[int, ...]) -> float:
    return 1.0 / math.sqrt(math.prod(normalized_shape))


class _FixedUnitNormAffine(nn.Module):
    """Shared state-dict and affine-buffer behavior for constrained norms."""

    normalized_shape: Tuple[int, ...]
    weight: Tensor
    bias: Optional[Tensor]

    def _init_fixed_affine(
        self,
        normalized_shape: NormalizedShape,
        *,
        bias: bool,
        device=None,
        dtype=None,
    ) -> None:
        self.normalized_shape = _canonical_shape(normalized_shape)
        self.affine_scale = _affine_scale(self.normalized_shape)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.register_buffer(
            "weight",
            torch.full(self.normalized_shape, self.affine_scale, **factory_kwargs),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(self.normalized_shape, **factory_kwargs),
            )
        else:
            self.register_buffer("bias", None)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Restore the exact constrained affine values."""

        self.weight.fill_(self.affine_scale)
        if self.bias is not None:
            self.bias.zero_()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # Consume standard normalization entries for checkpoint compatibility,
        # then restore the hard constraint.
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # The affine is deterministic, so checkpoints from normalization
        # implementations with no weight or bias can omit these keys.
        for name in ("weight", "bias"):
            key = prefix + name
            while key in missing_keys:
                missing_keys.remove(key)
        self.reset_parameters()


class UnitNormLayerNorm(_FixedUnitNormAffine):
    """LayerNorm with fixed ``gamma=1/sqrt(d)`` and ``beta=0``."""

    def __init__(
        self,
        normalized_shape: NormalizedShape,
        eps: float = 1e-5,
        *,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if eps < 0:
            raise ValueError(f"eps must be nonnegative, got {eps}")
        self.eps = float(eps)
        self.elementwise_affine = True
        self._init_fixed_affine(
            normalized_shape,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        accumulation_dtype = (
            torch.float32
            if input_dtype in (torch.float16, torch.bfloat16)
            else input_dtype
        )
        normalized = F.layer_norm(
            hidden_states.to(accumulation_dtype),
            self.normalized_shape,
            weight=None,
            bias=None,
            eps=self.eps,
        )
        return (normalized * self.affine_scale).to(input_dtype)

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"affine_scale={self.affine_scale:.8g}, bias={self.bias is not None}"
        )


class UnitNormRMSNorm(_FixedUnitNormAffine):
    """RMSNorm with fixed ``gamma=1/sqrt(d)`` and optional fixed zero bias."""

    def __init__(
        self,
        normalized_shape: NormalizedShape,
        eps: Optional[float] = 1e-6,
        *,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if eps is not None and eps < 0:
            raise ValueError(f"eps must be nonnegative or None, got {eps}")
        self.eps = None if eps is None else float(eps)
        # Several Hugging Face implementations use this attribute name.
        self.variance_epsilon = self.eps
        self.elementwise_affine = True
        self._init_fixed_affine(
            normalized_shape,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim < len(self.normalized_shape):
            raise RuntimeError(
                f"expected at least {len(self.normalized_shape)} dimensions, "
                f"got input shape {tuple(hidden_states.shape)}"
            )
        actual_shape = tuple(hidden_states.shape[-len(self.normalized_shape) :])
        if actual_shape != self.normalized_shape:
            raise RuntimeError(
                f"expected trailing dimensions {self.normalized_shape}, "
                f"got input shape {tuple(hidden_states.shape)}"
            )

        input_dtype = hidden_states.dtype
        accumulation_dtype = (
            torch.float32
            if input_dtype in (torch.float16, torch.bfloat16)
            else input_dtype
        )
        normalized = hidden_states.to(accumulation_dtype)
        dimensions = tuple(range(-len(self.normalized_shape), 0))
        variance = normalized.pow(2).mean(dim=dimensions, keepdim=True)
        eps = self.eps
        if eps is None:
            eps = torch.finfo(input_dtype).eps
        normalized = normalized * torch.rsqrt(variance + eps)
        return (normalized * self.affine_scale).to(input_dtype)

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"affine_scale={self.affine_scale:.8g}, bias={self.bias is not None}"
        )


def _local_device_and_dtype(module: nn.Module):
    tensors = list(module.parameters(recurse=False)) + list(
        module.buffers(recurse=False)
    )
    for tensor in tensors:
        if tensor is not None:
            return tensor.device, tensor.dtype
    return None, None


def _is_rms_norm(module: nn.Module) -> bool:
    class_name = module.__class__.__name__.replace("_", "").lower()
    if "rmsnorm" in class_name:
        return True
    # T5LayerNorm and compatible implementations are RMSNorm despite their
    # names: they expose variance_epsilon, a 1D weight, and no bias.
    weight = getattr(module, "weight", None)
    return (
        hasattr(module, "variance_epsilon")
        and weight is not None
        and getattr(weight, "ndim", None) == 1
        and getattr(module, "bias", None) is None
    )


def _is_layer_norm(module: nn.Module) -> bool:
    if isinstance(module, nn.LayerNorm):
        return True
    class_name = module.__class__.__name__.replace("_", "").lower()
    return (
        class_name.endswith("layernorm")
        and not _is_rms_norm(module)
        and getattr(module, "normalized_shape", None) is not None
    )


def constrained_normalization(module: nn.Module) -> nn.Module:
    """Return the constrained equivalent of one LayerNorm or RMSNorm."""

    if isinstance(module, (UnitNormLayerNorm, UnitNormRMSNorm)):
        return module

    device, dtype = _local_device_and_dtype(module)
    if _is_layer_norm(module):
        replacement = UnitNormLayerNorm(
            module.normalized_shape,
            eps=getattr(module, "eps", 1e-5),
            bias=getattr(module, "bias", None) is not None,
            device=device,
            dtype=dtype,
        )
    elif _is_rms_norm(module):
        weight = getattr(module, "weight", None)
        normalized_shape = getattr(module, "normalized_shape", None)
        if normalized_shape is None and weight is not None:
            normalized_shape = tuple(weight.shape)
        if normalized_shape is None:
            raise TypeError(
                f"cannot infer normalized shape for {module.__class__.__name__}"
            )
        if hasattr(module, "variance_epsilon"):
            eps = module.variance_epsilon
        else:
            eps = getattr(module, "eps", 1e-6)
        replacement = UnitNormRMSNorm(
            normalized_shape,
            eps=eps,
            bias=getattr(module, "bias", None) is not None,
            device=device,
            dtype=dtype,
        )
    else:
        raise TypeError(
            f"expected a LayerNorm or RMSNorm module, got {module.__class__.__name__}"
        )

    replacement.train(module.training)
    return replacement


def _set_module_by_name(model: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent._modules[part]
    parent._modules[parts[-1]] = replacement


def apply_normalization_affine_constraints(model: nn.Module) -> int:
    """Constrain all descendant LayerNorm/RMSNorm affine transformations.

    Apply this after loading pretrained weights and before constructing an
    optimizer. Every affine scale is fixed to ``1/sqrt(d)`` and every available
    bias is fixed to zero. The return value is the number of distinct
    normalization modules replaced. Repeated calls are idempotent.
    """

    replacements = {}
    replaced_ids = set()
    for module_name, module in list(model.named_modules(remove_duplicate=False)):
        if not module_name or isinstance(module, (UnitNormLayerNorm, UnitNormRMSNorm)):
            continue
        if not _is_layer_norm(module) and not _is_rms_norm(module):
            continue
        module_id = id(module)
        if module_id not in replacements:
            replacements[module_id] = constrained_normalization(module)
            replaced_ids.add(module_id)
        _set_module_by_name(model, module_name, replacements[module_id])
    return len(replaced_ids)


__all__ = [
    "UnitNormLayerNorm",
    "UnitNormRMSNorm",
    "apply_normalization_affine_constraints",
    "constrained_normalization",
]
