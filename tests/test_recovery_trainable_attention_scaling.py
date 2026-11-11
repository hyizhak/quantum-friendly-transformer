from argparse import Namespace

import pytest
import torch
from torch import nn

from quantum_friendly_transformer.experiments.runners.recovery.glue_sequence_classifier import (
    TRAIN_SCOPE_CHOICES,
    calibrate_attention_input_bound,
    configure_trainable,
    make_optimizer,
    validate_attention_scaling_args,
    warmup_attention_scales,
)
from quantum_friendly_transformer.norm_transformer.attention_scaling import (
    LearnableAttentionScale,
    apply_attention_scaling,
    attention_scale_native_regularization,
)


class TinyRecoveryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 2)
        self.attention = nn.Module()
        self.attention.add_module(
            "_qft_learnable_attention_scale",
            LearnableAttentionScale(1e-4),
        )


@pytest.mark.parametrize("train_scope", TRAIN_SCOPE_CHOICES)
def test_scale_parameter_is_enabled_for_every_train_scope(train_scope):
    model = TinyRecoveryModel()

    configure_trainable(
        model,
        train_scope,
        capped=False,
        trainable_attention_scaling=True,
    )

    assert model.attention._qft_learnable_attention_scale.raw_scale.requires_grad


def test_scale_parameter_is_enabled_after_peft_freeze():
    model = TinyRecoveryModel()

    configure_trainable(
        model,
        "head_caps",
        capped=True,
        peft_lora=True,
        trainable_attention_scaling=True,
    )

    assert model.attention._qft_learnable_attention_scale.raw_scale.requires_grad


def test_optimizer_uses_dedicated_zero_decay_scale_group():
    model = TinyRecoveryModel()
    configure_trainable(
        model,
        "all",
        capped=False,
        trainable_attention_scaling=True,
    )
    trainable = [param for param in model.parameters() if param.requires_grad]
    scale_params = [model.attention._qft_learnable_attention_scale.raw_scale]
    args = Namespace(
        optimizer="adamw",
        learning_rate=2e-4,
        weight_decay=0.01,
        attention_scale_learning_rate=1e-2,
        attention_scale_adam_eps=1e-12,
    )

    optimizer = make_optimizer(args, trainable, scale_params)
    groups = {group["group_name"]: group for group in optimizer.param_groups}

    assert groups["model"]["lr"] == pytest.approx(2e-4)
    assert groups["model"]["weight_decay"] == pytest.approx(0.01)
    assert groups["attention_scale"]["lr"] == pytest.approx(1e-2)
    assert groups["attention_scale"]["weight_decay"] == 0
    assert groups["attention_scale"]["eps"] == pytest.approx(1e-12)


def test_trainable_scaling_validation_rejects_non_alpha_mode():
    args = Namespace(
        attention_scaling="native",
        alpha_s=None,
        alpha_w=None,
        trainable_attention_scaling=True,
        device_map=None,
    )

    with pytest.raises(ValueError, match="requires.*alpha_s_alpha_w"):
        validate_attention_scaling_args(args)


def _bounded_attention_args(**overrides):
    values = {
        "attention_scaling": "alpha_s_alpha_w",
        "alpha_s": 2.0,
        "alpha_w": 3.0,
        "trainable_attention_scaling": True,
        "shared_attention_scale": True,
        "enforce_alpha_denominator_bound": True,
        "device_map": None,
        "alpha_s_calibration_batches": 0,
        "alpha_s_calibration_margin": 1.0,
        "attention_scale_learning_rate": 1e-2,
        "attention_scale_adam_eps": 1e-12,
        "attention_scale_min": 1e-8,
        "attention_scale_max": 2.0,
        "attention_scale_native_regularization": 0.0,
        "attention_scale_prior_warmup_steps": 0,
        "attention_scale_prior_warmup_learning_rate": 0.3,
        "attention_scale_prior_warmup_snap_native": False,
        "freeze_attention_scale_after_prior_warmup": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_bounded_attention_scaling_validation_accepts_strict_configuration():
    validate_attention_scaling_args(_bounded_attention_args())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "trainable_attention_scaling": False,
                "shared_attention_scale": False,
            },
            "requires trainable.*alpha_s_alpha_w",
        ),
        (
            {
                "attention_scale_prior_warmup_steps": 1,
                "attention_scale_native_regularization": 1.0,
            },
            "incompatible with native-prior warmup",
        ),
        (
            {"attention_scale_native_regularization": 1.0},
            "incompatible with native-scale regularization",
        ),
        (
            {"attention_scale_prior_warmup_snap_native": True},
            "incompatible with snapping.*native",
        ),
    ],
)
def test_bounded_attention_scaling_validation_rejects_bound_relaxation(
    overrides,
    message,
):
    with pytest.raises(ValueError, match=message):
        validate_attention_scaling_args(_bounded_attention_args(**overrides))


class GPTNeoSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 4
        self.q_proj = nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states):
        return self.q_proj(hidden_states)


class TinyWarmupModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = GPTNeoSelfAttention()


class TinyCalibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = GPTNeoSelfAttention()

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        return self.attention(hidden_states=input_ids.float())


def test_alpha_s_calibration_preserves_training_rng_state():
    from torch.utils.data import DataLoader

    model = TinyCalibrationModel()
    loader = DataLoader(
        [
            {
                "input_ids": torch.ones(3, 4),
                "attention_mask": torch.ones(3),
                "labels": torch.tensor(0),
            }
            for _ in range(4)
        ],
        batch_size=2,
        shuffle=False,
    )
    torch.manual_seed(1234)
    state_before = torch.random.get_rng_state().clone()

    result = calibrate_attention_input_bound(
        torch,
        model,
        loader,
        torch.device("cpu"),
        max_batches=2,
    )

    assert torch.equal(torch.random.get_rng_state(), state_before)
    assert result["batches"] == 2
    assert result["observed_max_token_l2"] > 0
    assert model.training


def test_controller_only_warmup_moves_scale_without_changing_model_weights():
    model = TinyWarmupModel()
    apply_attention_scaling(
        model,
        "alpha_s_alpha_w",
        alpha_s=20.0,
        alpha_w=24.0,
        trainable=True,
        shared_trainable=True,
    )
    controller = model._qft_learnable_attention_scale
    base_weight = model.attention.q_proj.weight.detach().clone()
    initial_loss = attention_scale_native_regularization(model).item()
    args = Namespace(
        attention_scale_prior_warmup_steps=25,
        attention_scale_prior_warmup_learning_rate=0.3,
        attention_scale_prior_warmup_snap_native=False,
        attention_scale_adam_eps=1e-12,
    )

    result = warmup_attention_scales(torch, model, [controller.raw_scale], args)

    assert result["best_regularization"] < initial_loss
    assert controller.score_scale > controller.initial_score_scale
    torch.testing.assert_close(model.attention.q_proj.weight, base_weight)
