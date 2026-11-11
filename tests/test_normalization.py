import math

import torch
from torch import nn

from quantum_friendly_transformer.norm_transformer.normalization import (
    UnitNormLayerNorm,
    UnitNormRMSNorm,
    apply_normalization_affine_constraints,
    constrained_normalization,
)


class LlamaRMSNorm(nn.Module):
    """Small stand-in for Hugging Face RMSNorm implementations."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return (self.weight.float() * normalized).to(hidden_states.dtype)


class OlmoLayerNorm(nn.Module):
    """Small stand-in for OLMo's parameter-free LayerNorm."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.normalized_shape = (hidden_size,)

    def forward(self, hidden_states):
        return nn.functional.layer_norm(
            hidden_states,
            self.normalized_shape,
            weight=None,
            bias=None,
            eps=1e-5,
        )


def test_unit_norm_layer_norm_has_fixed_affine_and_unit_output():
    norm = UnitNormLayerNorm(4, eps=0.0)
    hidden_states = torch.tensor(
        [[[1.0, 2.0, 4.0, 8.0], [-3.0, 1.0, 2.0, 6.0]]]
    )

    output = norm(hidden_states)

    torch.testing.assert_close(norm.weight, torch.full((4,), 0.5))
    torch.testing.assert_close(norm.bias, torch.zeros(4))
    torch.testing.assert_close(
        torch.linalg.vector_norm(output, dim=-1),
        torch.ones(1, 2),
    )
    assert list(norm.parameters()) == []


def test_unit_norm_rms_norm_has_fixed_affine_and_unit_output():
    norm = UnitNormRMSNorm(9, eps=0.0)
    hidden_states = torch.arange(1.0, 19.0).reshape(2, 9)

    output = norm(hidden_states)

    torch.testing.assert_close(norm.weight, torch.full((9,), 1.0 / 3.0))
    torch.testing.assert_close(
        torch.linalg.vector_norm(output, dim=-1),
        torch.ones(2),
    )
    assert list(norm.parameters()) == []


def test_normalized_dimension_is_product_of_normalized_shape():
    norm = UnitNormLayerNorm((2, 8), eps=0.0)
    assert norm.affine_scale == 0.25
    torch.testing.assert_close(norm.weight, torch.full((2, 8), 0.25))


def test_checkpoint_load_cannot_overwrite_constraint():
    norm = UnitNormLayerNorm(4)
    norm.load_state_dict(
        {
            "weight": torch.full((4,), 7.0),
            "bias": torch.full((4,), 11.0),
        }
    )

    torch.testing.assert_close(norm.weight, torch.full((4,), 1.0 / math.sqrt(4)))
    torch.testing.assert_close(norm.bias, torch.zeros(4))


def test_apply_constraints_replaces_layer_norm_and_rms_norm_idempotently():
    shared_layer_norm = nn.LayerNorm(4)
    model = nn.ModuleDict(
        {
            "layer_norm": shared_layer_norm,
            "shared_layer_norm": shared_layer_norm,
            "rms_norm": LlamaRMSNorm(4),
            "parameter_free_layer_norm": OlmoLayerNorm(4),
            "linear": nn.Linear(4, 4),
        }
    )
    model.eval()
    original_state = model.state_dict()
    linear = model["linear"]

    replaced = apply_normalization_affine_constraints(model)

    assert replaced == 3
    assert isinstance(model["layer_norm"], UnitNormLayerNorm)
    assert model["layer_norm"] is model["shared_layer_norm"]
    assert isinstance(model["rms_norm"], UnitNormRMSNorm)
    assert isinstance(model["parameter_free_layer_norm"], UnitNormLayerNorm)
    assert model["linear"] is linear
    assert not model["layer_norm"].training
    model.load_state_dict(original_state)
    assert apply_normalization_affine_constraints(model) == 0


def test_constrained_normalization_supports_a_root_module():
    constrained = constrained_normalization(nn.LayerNorm(16, bias=False))
    assert isinstance(constrained, UnitNormLayerNorm)
    assert constrained.bias is None
    torch.testing.assert_close(constrained.weight, torch.full((16,), 0.25))
