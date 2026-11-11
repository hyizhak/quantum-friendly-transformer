import math

import pytest
import torch
from torch import nn

from quantum_friendly_transformer.norm_transformer.attention_scaling import (
    AlphaBoundedAttentionScale,
    LearnableAttentionScale,
    apply_attention_scaling,
    attention_scale_native_regularization,
    attention_query_scale,
    collect_attention_scaling_stats,
    project_attention_scales_,
    set_attention_scales_to_native_,
)


class GPTNeoSelfAttention(nn.Module):
    """Small stand-in with the public attributes used by the adapter."""

    def __init__(self, hidden_size: int = 8, num_heads: int = 2):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class TinyGPTNeo(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = GPTNeoSelfAttention()


class TwoLayerTinyGPTNeo(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = GPTNeoSelfAttention()
        self.attn2 = GPTNeoSelfAttention()


class OlmoAttention(nn.Module):
    """Stand-in for backbones that expose their raw-score multiplier."""

    def __init__(self, hidden_size: int = 8, num_heads: int = 2):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scaling = 1 / math.sqrt(self.head_dim)

    def forward(self, hidden_states):
        return self.q_proj(hidden_states) * self.scaling


class TinyOlmo(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = OlmoAttention()


def test_attention_query_scale():
    assert attention_query_scale("native", head_dim=4) == 1.0
    assert attention_query_scale("sqrt_head_dim", head_dim=4) == 0.5
    assert attention_query_scale("sequence_length", head_dim=4, sequence_length=128) == 1 / 128
    assert attention_query_scale(
        "alpha_s_alpha_w",
        head_dim=4,
        alpha_s=2.0,
        alpha_w=3.0,
    ) == 1 / 36


@pytest.mark.parametrize(
    ("scaling", "sequence_length", "expected"),
    [
        ("sqrt_head_dim", None, 1 / math.sqrt(4)),
        ("sequence_length", 8, 1 / 8),
    ],
)
def test_apply_attention_scaling_scales_queries(scaling, sequence_length, expected):
    model = TinyGPTNeo()
    with torch.no_grad():
        model.attn.q_proj.weight.copy_(torch.eye(8))

    rows = apply_attention_scaling(model, scaling, sequence_length=sequence_length)
    output = model.attn.q_proj(torch.ones(2, 3, 8))

    assert len(rows) == 1
    assert rows[0]["query_scale"] == expected
    torch.testing.assert_close(output, torch.full_like(output, expected))


def test_native_scaling_does_not_patch_model():
    model = TinyGPTNeo()
    assert apply_attention_scaling(model, "native") == []
    assert not hasattr(model.attn, "_qft_attention_scaling")


def test_sequence_length_scaling_requires_positive_n():
    with pytest.raises(ValueError, match="sequence_length must be positive"):
        attention_query_scale("sequence_length", head_dim=4)


@pytest.mark.parametrize(
    ("alpha_s", "alpha_w", "message"),
    [
        (None, 2.0, "alpha_s"),
        (0.0, 2.0, "alpha_s"),
        (2.0, None, "alpha_w"),
        (2.0, float("inf"), "alpha_w"),
        (1e308, 1e308, "must be finite"),
    ],
)
def test_alpha_scaling_requires_positive_finite_bounds(alpha_s, alpha_w, message):
    with pytest.raises(ValueError, match=message):
        attention_query_scale(
            "alpha_s_alpha_w",
            head_dim=4,
            alpha_s=alpha_s,
            alpha_w=alpha_w,
        )


def test_apply_alpha_scaling_scales_queries_and_records_denominator():
    model = TinyGPTNeo()
    with torch.no_grad():
        model.attn.q_proj.weight.copy_(torch.eye(8))

    rows = apply_attention_scaling(
        model,
        "alpha_s_alpha_w",
        alpha_s=2.0,
        alpha_w=3.0,
    )
    output = model.attn.q_proj(torch.ones(2, 3, 8))

    assert rows[0]["alpha_s"] == 2.0
    assert rows[0]["alpha_w"] == 3.0
    assert rows[0]["score_denominator"] == 36.0
    assert rows[0]["score_scale"] == 1 / 36
    torch.testing.assert_close(output, torch.full_like(output, 1 / 36))


def test_alpha_scaling_replaces_a_backends_native_score_scale():
    model = TinyOlmo()

    rows = apply_attention_scaling(
        model,
        "alpha_s_alpha_w",
        alpha_s=2.0,
        alpha_w=3.0,
    )

    assert rows[0]["adapter"] == "module_scaling"
    assert rows[0]["query_scale"] == 1.0
    assert model.attn.scaling == 1 / 36


def test_trainable_query_scale_has_gradients_and_round_trips_state_dict():
    model = TinyGPTNeo()
    rows = apply_attention_scaling(
        model,
        "alpha_s_alpha_w",
        alpha_s=2.0,
        alpha_w=3.0,
        trainable=True,
    )
    controller = model.attn._qft_learnable_attention_scale

    assert isinstance(controller, LearnableAttentionScale)
    assert rows[0]["adapter"] == "trainable_query_projection_hook"
    assert rows[0]["trainable"] is True
    assert "attn._qft_learnable_attention_scale.raw_scale" in model.state_dict()

    output = model.attn.q_proj(torch.ones(2, 3, 8))
    output.sum().backward()
    assert controller.raw_scale.grad is not None
    assert controller.raw_scale.grad.abs().item() > 0

    saved = model.state_dict()
    restored = TinyGPTNeo()
    apply_attention_scaling(
        restored,
        "alpha_s_alpha_w",
        alpha_s=2.0,
        alpha_w=3.0,
        trainable=True,
    )
    restored.load_state_dict(saved)
    torch.testing.assert_close(
        restored.attn._qft_learnable_attention_scale.score_scale,
        controller.score_scale,
    )


def test_trainable_module_scale_updates_on_each_forward():
    model = TinyOlmo()
    rows = apply_attention_scaling(
        model,
        "alpha_s_alpha_w",
        alpha_s=2.0,
        alpha_w=3.0,
        trainable=True,
    )
    controller = model.attn._qft_learnable_attention_scale
    with torch.no_grad():
        initial = controller.score_scale.detach().clone()
        controller.raw_scale.add_(1.0)

    model.attn(torch.ones(1, 2, 8))
    assert model.attn.scaling > initial
    stats = collect_attention_scaling_stats(model)

    assert rows[0]["adapter"] == "trainable_module_scaling"
    assert len(stats) == 1
    assert stats[0]["score_scale"] > 1 / 36
    assert stats[0]["score_denominator"] < 36.0
    assert stats[0]["effective_alpha_product"] == pytest.approx(
        math.sqrt(stats[0]["score_denominator"])
    )


def test_trainable_scale_is_bounded_and_initialized_exactly():
    controller = LearnableAttentionScale(
        1 / 36,
        min_score_scale=1e-8,
        max_score_scale=2.0,
    )

    assert controller.score_scale.item() == pytest.approx(1 / 36, rel=1e-6)
    with torch.no_grad():
        controller.raw_scale.fill_(-100)
    assert controller.score_scale.item() == pytest.approx(1e-8, rel=1e-6)
    with torch.no_grad():
        controller.raw_scale.fill_(100)
    assert controller.score_scale.item() == pytest.approx(2.0, rel=1e-6)


def test_trainable_scale_rejects_non_eager_attention():
    model = TinyOlmo()
    model.config = type("Config", (), {"_attn_implementation": "sdpa"})()
    with pytest.raises(ValueError, match="requires eager attention"):
        apply_attention_scaling(
            model,
            "alpha_s_alpha_w",
            alpha_s=2.0,
            alpha_w=3.0,
            trainable=True,
        )


def test_shared_trainable_scale_has_one_parameter_and_round_trips():
    model = TwoLayerTinyGPTNeo()
    rows = apply_attention_scaling(
        model,
        "alpha_s_alpha_w",
        alpha_s=2.0,
        alpha_w=3.0,
        trainable=True,
        shared_trainable=True,
    )
    controller = model._qft_learnable_attention_scale

    assert len(rows) == 2
    assert all(row["shared_trainable"] for row in rows)
    assert not hasattr(model.attn1, "_qft_learnable_attention_scale")
    assert not hasattr(model.attn2, "_qft_learnable_attention_scale")
    assert [
        name
        for name in model.state_dict()
        if name.endswith("_qft_learnable_attention_scale.raw_scale")
    ] == ["_qft_learnable_attention_scale.raw_scale"]

    output = model.attn1.q_proj(torch.ones(1, 2, 8))
    output = output + model.attn2.q_proj(torch.ones(1, 2, 8))
    output.sum().backward()
    assert controller.raw_scale.grad is not None
    assert controller.raw_scale.grad.abs().item() > 0
    stats = collect_attention_scaling_stats(model)
    assert len(stats) == 1
    assert stats[0]["name"] == "<shared>"
    assert stats[0]["shared"] is True

    restored = TwoLayerTinyGPTNeo()
    apply_attention_scaling(
        restored,
        "alpha_s_alpha_w",
        alpha_s=2.0,
        alpha_w=3.0,
        trainable=True,
        shared_trainable=True,
    )
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(
        restored._qft_learnable_attention_scale.score_scale,
        controller.score_scale,
    )


def test_shared_scale_requires_trainable_alpha_mode():
    with pytest.raises(ValueError, match="requires trainable"):
        apply_attention_scaling(
            TinyGPTNeo(),
            "alpha_s_alpha_w",
            alpha_s=2.0,
            alpha_w=3.0,
            shared_trainable=True,
        )


def test_native_scale_regularization_has_gradient_toward_gptneo_native_scale():
    model = TwoLayerTinyGPTNeo()
    apply_attention_scaling(
        model,
        "alpha_s_alpha_w",
        alpha_s=2.0,
        alpha_w=3.0,
        trainable=True,
        shared_trainable=True,
    )
    controller = model._qft_learnable_attention_scale
    before = controller.score_scale.detach().clone()
    regularization = attention_scale_native_regularization(model)

    assert regularization.item() == pytest.approx(math.log(1 / 36) ** 2, rel=1e-5)
    regularization.backward()
    assert controller.raw_scale.grad.item() < 0
    optimizer = torch.optim.SGD([controller.raw_scale], lr=0.01)
    optimizer.step()
    assert controller.score_scale > before


def test_controller_can_be_set_exactly_to_native_scale():
    model = TwoLayerTinyGPTNeo()
    apply_attention_scaling(
        model,
        "alpha_s_alpha_w",
        alpha_s=20.0,
        alpha_w=24.0,
        trainable=True,
        shared_trainable=True,
    )

    set_attention_scales_to_native_(model)

    assert model._qft_learnable_attention_scale.score_scale.item() == pytest.approx(
        1.0,
        rel=1e-6,
    )
    assert attention_scale_native_regularization(model).item() == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_native_scaling_cannot_be_trainable():
    with pytest.raises(ValueError, match="requires an explicit non-native"):
        apply_attention_scaling(TinyGPTNeo(), "native", trainable=True)


def test_alpha_bounded_scale_starts_on_safe_boundary_and_moves_inward():
    alpha_score_scale = 1 / 36
    controller = AlphaBoundedAttentionScale(
        alpha_score_scale,
        min_score_scale=1e-8,
    )

    # The persistent float32 cap must be rounded toward safety, not to the
    # nearest float on the forbidden side of the mathematical boundary.
    assert controller.score_scale.item() == controller.alpha_score_scale.item()
    assert controller.score_scale.item() <= alpha_score_scale
    assert 1.0 / controller.score_scale.item() >= 36.0
    assert controller.log_denominator_slack.item() == 0.0

    controller.score_scale.backward()
    assert controller.log_denominator_slack.grad.item() < 0
    optimizer = torch.optim.SGD([controller.log_denominator_slack], lr=1.0)
    optimizer.step()

    assert controller.log_denominator_slack.item() > 0
    assert controller.score_scale.item() < controller.alpha_score_scale.item()
    assert 1.0 / controller.score_scale.item() > 36.0


def test_alpha_bound_survives_outward_adam_pressure_and_can_reverse_inward():
    model = TinyGPTNeo()
    apply_attention_scaling(
        model,
        "alpha_s_alpha_w",
        alpha_s=2.0,
        alpha_w=3.0,
        trainable=True,
        shared_trainable=True,
        enforce_alpha_denominator_bound=True,
        min_score_scale=1e-8,
    )
    controller = model._qft_learnable_attention_scale
    parameter = controller.log_denominator_slack
    optimizer = torch.optim.AdamW([parameter], lr=0.5, weight_decay=0.0)

    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        (-controller.score_scale).backward()
        assert parameter.grad.item() > 0
        optimizer.step()

        # The forward parameterization is a fail-safe even before projection.
        assert controller.score_scale.item() <= 1 / 36
        project_attention_scales_(model, optimizer)
        assert parameter.item() >= 0
        assert controller.score_scale.item() <= 1 / 36

    # Projection must remove outward Adam momentum. One genuinely inward
    # gradient can therefore leave the boundary immediately.
    optimizer.zero_grad(set_to_none=True)
    controller.score_scale.backward()
    assert parameter.grad.item() < 0
    optimizer.step()
    project_attention_scales_(model, optimizer)

    assert parameter.item() > 0
    assert controller.score_scale.item() < controller.alpha_score_scale.item()


def test_alpha_bounded_scale_rejects_native_snap_above_cap():
    model = TinyGPTNeo()
    apply_attention_scaling(
        model,
        "alpha_s_alpha_w",
        alpha_s=2.0,
        alpha_w=3.0,
        trainable=True,
        enforce_alpha_denominator_bound=True,
        min_score_scale=1e-8,
    )

    with pytest.raises(ValueError, match="(?:alpha|bound|ceiling)"):
        set_attention_scales_to_native_(model)

    assert model.attn._qft_learnable_attention_scale.score_scale.item() <= 1 / 36


def test_alpha_bounded_scale_shared_state_round_trip_and_stats():
    model = TwoLayerTinyGPTNeo()
    rows = apply_attention_scaling(
        model,
        "alpha_s_alpha_w",
        alpha_s=2.0,
        alpha_w=3.0,
        trainable=True,
        shared_trainable=True,
        enforce_alpha_denominator_bound=True,
        min_score_scale=1e-8,
    )
    controller = model._qft_learnable_attention_scale
    with torch.no_grad():
        controller.log_denominator_slack.fill_(0.75)

    assert isinstance(controller, AlphaBoundedAttentionScale)
    assert len(rows) == 2
    assert all(row["enforce_alpha_denominator_bound"] for row in rows)
    assert all(row["score_scale_ceiling"] == pytest.approx(1 / 36) for row in rows)
    assert all(row["score_denominator_floor"] == pytest.approx(36.0) for row in rows)
    assert [
        name
        for name in model.state_dict()
        if name.endswith(".log_denominator_slack")
    ] == ["_qft_learnable_attention_scale.log_denominator_slack"]
    assert (
        "_qft_learnable_attention_scale.alpha_score_scale" in model.state_dict()
    )

    stats = collect_attention_scaling_stats(model)
    assert len(stats) == 1
    assert stats[0]["name"] == "<shared>"
    assert stats[0]["shared"] is True
    assert stats[0]["score_scale"] <= stats[0]["score_scale_ceiling"]
    assert stats[0]["score_denominator"] >= stats[0]["score_denominator_floor"]
    assert stats[0]["score_denominator_floor"] >= 36.0

    restored = TwoLayerTinyGPTNeo()
    apply_attention_scaling(
        restored,
        "alpha_s_alpha_w",
        alpha_s=2.0,
        alpha_w=3.0,
        trainable=True,
        shared_trainable=True,
        enforce_alpha_denominator_bound=True,
        min_score_scale=1e-8,
    )
    restored.load_state_dict(model.state_dict(), strict=True)

    restored_controller = restored._qft_learnable_attention_scale
    torch.testing.assert_close(
        restored_controller.log_denominator_slack,
        controller.log_denominator_slack,
    )
    torch.testing.assert_close(
        restored_controller.alpha_score_scale,
        controller.alpha_score_scale,
    )
    torch.testing.assert_close(
        restored_controller.score_scale,
        controller.score_scale,
    )
    assert restored_controller.score_scale.item() <= 1 / 36


def test_alpha_bound_mode_requires_trainable_alpha_scaling():
    with pytest.raises(ValueError, match="requires trainable.*alpha_s_alpha_w"):
        apply_attention_scaling(
            TinyGPTNeo(),
            "alpha_s_alpha_w",
            alpha_s=2.0,
            alpha_w=3.0,
            enforce_alpha_denominator_bound=True,
        )
