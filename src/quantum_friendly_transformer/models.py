"""PyTorch-only reference models for sequence and token classification.

The original implementation modules remain available under ``norm_transformer``
for compatibility with existing imports and checkpoints. The optional BERT
models are exposed separately in ``quantum_friendly_transformer.norm_transformer.bert``.
"""

from .norm_transformer.frobenius_normalized_transformer import (
    FrobeniuslyNormalizedTransformerBlock,
    FrobeniuslyNormalizedTransformerForSequenceClassification,
    FrobeniuslyNormalizedTransformerForTokenClassification,
)
from .norm_transformer.spectral_normalized_transformer import (
    SpectrallyNormalizedTransformerBlock,
    SpectrallyNormalizedTransformerForSequenceClassification,
    SpectrallyNormalizedTransformerForTokenClassification,
)

__all__ = [
    "FrobeniuslyNormalizedTransformerBlock",
    "FrobeniuslyNormalizedTransformerForSequenceClassification",
    "FrobeniuslyNormalizedTransformerForTokenClassification",
    "SpectrallyNormalizedTransformerBlock",
    "SpectrallyNormalizedTransformerForSequenceClassification",
    "SpectrallyNormalizedTransformerForTokenClassification",
]
