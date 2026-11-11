"""Reusable PyTorch components for quantum-friendly Transformer research.

The core API needs only PyTorch. Hugging Face models and experiment runners
have separate optional dependencies; importing this package never loads a
pretrained model or starts an experiment.
"""

from ._version import __version__
from .norm_transformer import (
    ATTENTION_SCALING_CHOICES,
    AlphaBoundedAttentionScale,
    LearnableAttentionScale,
    LearnableFrobeniusCap,
    UnitNormLayerNorm,
    UnitNormRMSNorm,
    apply_attention_scaling,
    apply_learnable_frobenius_caps,
    apply_normalization_affine_constraints,
    attention_query_scale,
    attention_scale_native_regularization,
    collect_attention_scaling_stats,
    collect_learnable_cap_stats,
    collect_weight_stats,
    constrained_normalization,
    frobenius_norm,
    frobenius_norm_with_scaling,
    learnable_frobenius_cap,
    project_attention_scales_,
    set_attention_scales_to_native_,
)

__all__ = [
    "__version__",
    "ATTENTION_SCALING_CHOICES",
    "AlphaBoundedAttentionScale",
    "LearnableAttentionScale",
    "LearnableFrobeniusCap",
    "UnitNormLayerNorm",
    "UnitNormRMSNorm",
    "apply_attention_scaling",
    "apply_learnable_frobenius_caps",
    "apply_normalization_affine_constraints",
    "attention_query_scale",
    "attention_scale_native_regularization",
    "collect_attention_scaling_stats",
    "collect_learnable_cap_stats",
    "collect_weight_stats",
    "constrained_normalization",
    "frobenius_norm",
    "frobenius_norm_with_scaling",
    "learnable_frobenius_cap",
    "project_attention_scales_",
    "set_attention_scales_to_native_",
]
