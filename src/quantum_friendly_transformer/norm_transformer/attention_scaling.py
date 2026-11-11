"""Attention-score scaling adapters for pretrained transformer backbones.

This module makes the score denominator explicit for Hugging Face GPT-Neo,
OLMo, and Mistral backbones without replacing projection modules, so it
composes with the normalization parametrizations in :mod:`frobenius_cap`.
Fixed adapters do not change checkpoint keys. Trainable adapters add one small
registered controller per attention layer and must be reconstructed before a
strict state-dict load.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn


AttentionScaling = Literal[
    "native",
    "sqrt_head_dim",
    "sequence_length",
    "alpha_s_alpha_w",
]
ATTENTION_SCALING_CHOICES = (
    "native",
    "sqrt_head_dim",
    "sequence_length",
    "alpha_s_alpha_w",
)

DEFAULT_MIN_ATTENTION_SCALE = 1e-12
DEFAULT_MAX_ATTENTION_SCALE = 2.0


class LearnableAttentionScale(nn.Module):
    """Positive, bounded score multiplier initialized from fixed alpha bounds.

    The logarithm is optimized so the effective attention scale remains
    positive. A separate optimizer learning rate can move this scalar across
    the several orders of magnitude between a conservative quantum bound and
    a pretrained backbone's native attention temperature.
    """

    def __init__(
        self,
        initial_score_scale: float,
        *,
        min_score_scale: float = DEFAULT_MIN_ATTENTION_SCALE,
        max_score_scale: float = DEFAULT_MAX_ATTENTION_SCALE,
    ) -> None:
        super().__init__()
        if not math.isfinite(initial_score_scale) or initial_score_scale <= 0:
            raise ValueError("initial_score_scale must be a positive finite number")
        if not math.isfinite(min_score_scale) or min_score_scale <= 0:
            raise ValueError("min_score_scale must be a positive finite number")
        if not math.isfinite(max_score_scale) or max_score_scale <= min_score_scale:
            raise ValueError("max_score_scale must be finite and greater than min_score_scale")
        if not min_score_scale < initial_score_scale < max_score_scale:
            raise ValueError(
                "initial_score_scale must lie strictly between min_score_scale "
                "and max_score_scale"
            )
        log_min = math.log(min_score_scale)
        log_max = math.log(max_score_scale)
        initial_fraction = (math.log(initial_score_scale) - log_min) / (log_max - log_min)
        initial_raw = math.log(initial_fraction / (1.0 - initial_fraction))
        self.register_buffer(
            "initial_score_scale",
            torch.tensor(float(initial_score_scale), dtype=torch.float32),
        )
        self.register_buffer(
            "log_min_score_scale",
            torch.tensor(log_min, dtype=torch.float32),
        )
        self.register_buffer(
            "log_max_score_scale",
            torch.tensor(log_max, dtype=torch.float32),
        )
        self.raw_scale = nn.Parameter(torch.tensor(initial_raw, dtype=torch.float32))

    def forward(self) -> Tensor:
        fraction = self.raw_scale.sigmoid()
        log_scale = self.log_min_score_scale + fraction * (
            self.log_max_score_scale - self.log_min_score_scale
        )
        return log_scale.exp()

    @property
    def score_scale(self) -> Tensor:
        return self()

    @property
    def min_score_scale(self) -> Tensor:
        return self.log_min_score_scale.exp()

    @property
    def max_score_scale(self) -> Tensor:
        return self.log_max_score_scale.exp()

    @property
    def score_scale_ceiling(self) -> Tensor | None:
        return None

    @property
    def alpha_denominator_bound_enforced(self) -> bool:
        return False

    def _validated_raw_scale(self, score_scale: float) -> float:
        if not math.isfinite(score_scale) or score_scale <= 0:
            raise ValueError("score_scale must be a positive finite number")
        log_min = float(self.log_min_score_scale.detach().cpu().item())
        log_max = float(self.log_max_score_scale.detach().cpu().item())
        fraction = (math.log(score_scale) - log_min) / (log_max - log_min)
        if not 0.0 < fraction < 1.0:
            raise ValueError("score_scale must lie strictly inside the configured bounds")
        return math.log(fraction / (1.0 - fraction))

    def set_score_scale_(self, score_scale: float) -> None:
        """Set a representable score scale through the inverse parameterization."""

        raw = self._validated_raw_scale(score_scale)
        with torch.no_grad():
            self.raw_scale.fill_(raw)

    def project_score_scale_(self) -> bool:
        """Return false because this controller needs no explicit projection."""

        return False


def _conservative_float32(value: float, *, round_up: bool) -> Tensor:
    """Return a finite positive float32 rounded in the requested direction."""

    result = torch.tensor(float(value), dtype=torch.float32)
    if not bool(torch.isfinite(result).item()) or float(result.item()) <= 0:
        raise ValueError(f"{value:g} is not representable as a positive finite float32")
    represented = float(result.item())
    if (round_up and represented < value) or (not round_up and represented > value):
        direction = result.new_tensor(math.inf if round_up else 0.0)
        result = torch.nextafter(result, direction)
    return result


class AlphaBoundedAttentionScale(nn.Module):
    """Learn a denominator that may grow but never shrink below the alpha bound.

    With ``D_alpha = alpha_s**2 * alpha_w**2`` and ``s_alpha = 1 / D_alpha``,
    this controller parameterizes

    ``D(t) = D_alpha * exp(r(t))`` and ``s(t) = s_alpha * exp(-r(t))``

    for a projected log-denominator slack ``r(t) >= 0``. The forward clamp is a
    fail-safe that preserves ``s(t) <= s_alpha`` even between an optimizer step
    and projection. Projecting immediately after every optimizer step restores
    a usable boundary gradient and clears incompatible optimizer momentum.
    """

    CONSTRAINT_VERSION = 1

    def __init__(
        self,
        initial_score_scale: float,
        *,
        min_score_scale: float = DEFAULT_MIN_ATTENTION_SCALE,
        max_score_scale: float = DEFAULT_MAX_ATTENTION_SCALE,
    ) -> None:
        super().__init__()
        if not math.isfinite(initial_score_scale) or initial_score_scale <= 0:
            raise ValueError("initial_score_scale must be a positive finite number")
        if not math.isfinite(min_score_scale) or min_score_scale <= 0:
            raise ValueError("min_score_scale must be a positive finite number")
        if not math.isfinite(max_score_scale) or max_score_scale <= min_score_scale:
            raise ValueError("max_score_scale must be finite and greater than min_score_scale")
        if not min_score_scale < initial_score_scale < max_score_scale:
            raise ValueError(
                "initial_score_scale must lie strictly between min_score_scale "
                "and max_score_scale"
            )

        # Rounding the ceiling down makes the implemented float32 scale no
        # larger than the real-valued alpha scale. Rounding the numerical floor
        # up preserves the configured lower scale bound as well.
        alpha_score_scale = _conservative_float32(
            initial_score_scale,
            round_up=False,
        )
        minimum = _conservative_float32(min_score_scale, round_up=True)
        if not bool((minimum < alpha_score_scale).item()):
            raise ValueError(
                "min_score_scale must remain below the alpha score scale in float32"
            )
        max_slack = math.log(float(alpha_score_scale.item()) / float(minimum.item()))
        self.register_buffer("alpha_score_scale", alpha_score_scale)
        self.register_buffer("minimum_score_scale", minimum)
        self.register_buffer(
            "max_log_denominator_slack",
            torch.tensor(max_slack, dtype=torch.float32),
        )
        self.register_buffer(
            "constraint_version",
            torch.tensor(self.CONSTRAINT_VERSION, dtype=torch.int64),
        )
        self.register_buffer(
            "_projection_count",
            torch.tensor(0, dtype=torch.int64),
            persistent=False,
        )
        self.log_denominator_slack = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self) -> Tensor:
        slack = torch.clamp(
            self.log_denominator_slack,
            min=self.log_denominator_slack.new_zeros(()),
            max=self.max_log_denominator_slack,
        )
        score_scale = self.alpha_score_scale * torch.exp(-slack)
        return torch.clamp(
            score_scale,
            min=self.minimum_score_scale,
            max=self.alpha_score_scale,
        )

    @property
    def score_scale(self) -> Tensor:
        return self()

    @property
    def initial_score_scale(self) -> Tensor:
        return self.alpha_score_scale

    @property
    def min_score_scale(self) -> Tensor:
        return self.minimum_score_scale

    @property
    def max_score_scale(self) -> Tensor:
        return self.alpha_score_scale

    @property
    def score_scale_ceiling(self) -> Tensor:
        return self.alpha_score_scale

    @property
    def alpha_denominator_bound_enforced(self) -> bool:
        return True

    @property
    def projection_count(self) -> int:
        return int(self._projection_count.detach().cpu().item())

    def _validated_log_denominator_slack(self, score_scale: float) -> float:
        if not math.isfinite(score_scale) or score_scale <= 0:
            raise ValueError("score_scale must be a positive finite number")
        ceiling = float(self.alpha_score_scale.detach().cpu().item())
        minimum = float(self.minimum_score_scale.detach().cpu().item())
        if score_scale > ceiling:
            raise ValueError(
                f"score_scale cannot exceed the enforced alpha ceiling {ceiling:g}"
            )
        if score_scale < minimum:
            raise ValueError(
                f"score_scale cannot be below the configured minimum {minimum:g}"
            )
        return math.log(ceiling / score_scale)

    def set_score_scale_(self, score_scale: float) -> None:
        """Set a feasible score scale without relaxing the denominator bound."""

        slack = self._validated_log_denominator_slack(score_scale)
        with torch.no_grad():
            self.log_denominator_slack.fill_(slack)

    def project_score_scale_(self) -> bool:
        """Project raw log slack into its feasible interval."""

        with torch.no_grad():
            before = self.log_denominator_slack.detach().clone()
            self.log_denominator_slack.clamp_(
                min=0.0,
                max=float(self.max_log_denominator_slack.detach().cpu().item()),
            )
            projected = not bool(
                torch.equal(before, self.log_denominator_slack.detach())
            )
            if projected:
                self._projection_count.add_(1)
        return projected

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
        cap_key = prefix + "alpha_score_scale"
        saved_cap = state_dict.get(cap_key)
        if saved_cap is not None and not torch.equal(
            saved_cap.detach().cpu(),
            self.alpha_score_scale.detach().cpu(),
        ):
            error_msgs.append(
                f"alpha attention-scale ceiling mismatch for {cap_key}: "
                f"checkpoint={float(saved_cap.detach().float().cpu().item()):g}, "
                f"configured={float(self.alpha_score_scale.detach().cpu().item()):g}"
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self.project_score_scale_()


AttentionScaleController = LearnableAttentionScale | AlphaBoundedAttentionScale
_ATTENTION_SCALE_CONTROLLER_TYPES = (
    LearnableAttentionScale,
    AlphaBoundedAttentionScale,
)


def attention_query_scale(
    scaling: AttentionScaling,
    *,
    head_dim: int,
    sequence_length: int | None = None,
    alpha_s: float | None = None,
    alpha_w: float | None = None,
) -> float:
    """Return the multiplier that realizes the requested score denominator.

    In ``alpha_s_alpha_w`` mode this implements

    ``softmax((Q @ K.T) / (alpha_s**2 * alpha_w**2))``.

    The alpha values are fixed design bounds for a run. They are not inferred
    from the current batch or weights, which would make the training objective
    batch-dependent and add norm reductions to every forward pass.
    """

    if scaling not in ATTENTION_SCALING_CHOICES:
        choices = ", ".join(ATTENTION_SCALING_CHOICES)
        raise ValueError(f"unknown attention scaling {scaling!r}; expected one of: {choices}")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if scaling == "native":
        return 1.0
    if scaling == "sqrt_head_dim":
        return 1.0 / math.sqrt(head_dim)
    if scaling == "sequence_length":
        if sequence_length is None or sequence_length <= 0:
            raise ValueError("sequence_length must be positive for sequence_length scaling")
        return 1.0 / sequence_length
    if alpha_s is None or not math.isfinite(alpha_s) or alpha_s <= 0:
        raise ValueError("alpha_s must be a positive finite number for alpha_s_alpha_w scaling")
    if alpha_w is None or not math.isfinite(alpha_w) or alpha_w <= 0:
        raise ValueError("alpha_w must be a positive finite number for alpha_s_alpha_w scaling")
    try:
        denominator = alpha_s**2 * alpha_w**2
    except OverflowError as exc:
        raise ValueError("alpha_s**2 * alpha_w**2 must be finite") from exc
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("alpha_s**2 * alpha_w**2 must be a positive finite number")
    return 1.0 / denominator


def _attention_backend(module: nn.Module) -> str | None:
    # Avoid importing transformers at package-import time. Transformers is an
    # experiment dependency rather than a core library dependency.
    if (
        type(module).__name__ == "GPTNeoSelfAttention"
        and isinstance(getattr(module, "q_proj", None), nn.Module)
        and isinstance(getattr(module, "head_dim", None), int)
    ):
        return "gpt_neo"
    if (
        type(module).__name__ in {"MistralAttention", "OlmoAttention", "Olmo2Attention"}
        and isinstance(getattr(module, "q_proj", None), nn.Module)
        and isinstance(getattr(module, "head_dim", None), int)
        and isinstance(getattr(module, "scaling", None), float)
    ):
        return type(module).__name__[: -len("Attention")].lower()
    return None


def apply_attention_scaling(
    model: nn.Module,
    scaling: AttentionScaling,
    *,
    sequence_length: int | None = None,
    alpha_s: float | None = None,
    alpha_w: float | None = None,
    trainable: bool = False,
    shared_trainable: bool = False,
    enforce_alpha_denominator_bound: bool = False,
    min_score_scale: float = DEFAULT_MIN_ATTENTION_SCALE,
    max_score_scale: float = DEFAULT_MAX_ATTENTION_SCALE,
) -> list[dict]:
    """Apply an explicit attention-score denominator to supported layers.

    GPT-Neo receives a query-projection hook because its eager implementation
    uses raw scores. OLMo and Mistral expose the post-RoPE score scale directly,
    so the adapter updates that value. Fixed paths leave checkpoint keys
    unchanged. Trainable paths register controller state that must be
    reconstructed before loading a strict state dict.
    """

    # Validate even when native mode is selected, so misspellings never pass
    # silently. Native mode deliberately makes no model changes.
    attention_query_scale(
        scaling,
        head_dim=1,
        sequence_length=sequence_length,
        alpha_s=alpha_s,
        alpha_w=alpha_w,
    )
    if scaling == "native" and trainable:
        raise ValueError("trainable attention scaling requires an explicit non-native scaling mode")
    if shared_trainable and not trainable:
        raise ValueError("shared_trainable requires trainable attention scaling")
    if shared_trainable and scaling != "alpha_s_alpha_w":
        raise ValueError("shared trainable scaling is supported only for alpha_s_alpha_w mode")
    if enforce_alpha_denominator_bound and (
        not trainable or scaling != "alpha_s_alpha_w"
    ):
        raise ValueError(
            "enforce_alpha_denominator_bound requires trainable "
            "alpha_s_alpha_w scaling"
        )
    if scaling == "native":
        return []

    attention_implementation = getattr(
        getattr(model, "config", None), "_attn_implementation", None
    )
    if trainable and attention_implementation not in {None, "eager"}:
        raise ValueError(
            "trainable OLMo/Mistral attention scaling requires eager attention; "
            f"model is configured for {attention_implementation!r}"
        )

    targets = [
        (name, module, backend)
        for name, module in list(model.named_modules())
        if (backend := _attention_backend(module)) is not None
    ]
    if not targets:
        raise ValueError(
            "no supported attention layers found; explicit scaling currently supports "
            "Hugging Face GPT-Neo, OLMo, OLMo2, and Mistral attention backbones"
        )

    controller_class = (
        AlphaBoundedAttentionScale
        if enforce_alpha_denominator_bound
        else LearnableAttentionScale
    )
    shared_controller = None
    if shared_trainable:
        shared_scale = attention_query_scale(
            scaling,
            head_dim=targets[0][1].head_dim,
            sequence_length=sequence_length,
            alpha_s=alpha_s,
            alpha_w=alpha_w,
        )
        shared_controller = controller_class(
            shared_scale,
            min_score_scale=min_score_scale,
            max_score_scale=max_score_scale,
        )
        reference_parameter = next(targets[0][1].q_proj.parameters(), None)
        if reference_parameter is not None:
            shared_controller.to(device=reference_parameter.device)
        model.add_module("_qft_learnable_attention_scale", shared_controller)

    rows = []
    for name, module, backend in targets:
        if hasattr(module, "_qft_attention_scaling"):
            installed = getattr(module, "_qft_attention_scaling")
            raise RuntimeError(f"attention scaling is already installed on {name}: {installed}")

        head_dim = module.head_dim
        native_score_scale = 1.0 if backend == "gpt_neo" else float(module.scaling)
        score_scale = attention_query_scale(
            scaling,
            head_dim=head_dim,
            sequence_length=sequence_length,
            alpha_s=alpha_s,
            alpha_w=alpha_w,
        )
        scale_module = None
        if trainable:
            if shared_trainable:
                scale_module = shared_controller
            else:
                reference_parameter = next(module.q_proj.parameters(), None)
                scale_module = controller_class(
                    score_scale,
                    min_score_scale=min_score_scale,
                    max_score_scale=max_score_scale,
                )
                if reference_parameter is not None:
                    scale_module.to(device=reference_parameter.device)
                module.add_module("_qft_learnable_attention_scale", scale_module)

        if backend == "gpt_neo":
            query_scale = score_scale

            if trainable:

                def scale_query_trainable(
                    _module: nn.Module,
                    _inputs: tuple,
                    output: Tensor,
                    controller: AttentionScaleController = scale_module,
                ) -> Tensor:
                    return output * controller.score_scale

                module.q_proj.register_forward_hook(scale_query_trainable)
                adapter = "trainable_query_projection_hook"
            else:

                def scale_query(
                    _module: nn.Module,
                    _inputs: tuple,
                    output: Tensor,
                    scale: float = query_scale,
                ) -> Tensor:
                    return output * scale

                module.q_proj.register_forward_hook(scale_query)
                adapter = "query_projection_hook"
        else:
            if trainable:

                def update_score_scale(
                    attention_module: nn.Module,
                    _inputs: tuple,
                    controller: AttentionScaleController = scale_module,
                ) -> None:
                    attention_module.scaling = controller.score_scale

                module.register_forward_pre_hook(update_score_scale)
                adapter = "trainable_module_scaling"
            else:
                # Updating the native score scale is exact after Q/K
                # projection, clipping, and RoPE for eager, SDPA, and Flash
                # Attention paths.
                module.scaling = score_scale
                adapter = "module_scaling"
            query_scale = 1.0
        constrained_score_ceiling = (
            float(scale_module.score_scale_ceiling.detach().cpu().item())
            if isinstance(scale_module, AlphaBoundedAttentionScale)
            else None
        )
        installed_score_scale = (
            float(scale_module.score_scale.detach().cpu().item())
            if isinstance(scale_module, AlphaBoundedAttentionScale)
            else score_scale
        )
        module._qft_attention_scaling = {
            "scaling": scaling,
            "sequence_length": sequence_length,
            "alpha_s": alpha_s,
            "alpha_w": alpha_w,
            "trainable": trainable,
            "shared_trainable": shared_trainable,
            "enforce_alpha_denominator_bound": enforce_alpha_denominator_bound,
            "score_scale_ceiling": constrained_score_ceiling,
            "score_denominator_floor": (
                1.0 / constrained_score_ceiling
                if constrained_score_ceiling is not None
                else None
            ),
            "min_score_scale": min_score_scale if trainable else None,
            "max_score_scale": max_score_scale if trainable else None,
            "query_scale": query_scale,
            "native_score_scale": native_score_scale,
        }
        rows.append(
            {
                "name": name,
                "backend": backend,
                "adapter": adapter,
                "scaling": scaling,
                "head_dim": head_dim,
                "sequence_length": sequence_length,
                "alpha_s": alpha_s,
                "alpha_w": alpha_w,
                "trainable": trainable,
                "shared_trainable": shared_trainable,
                "enforce_alpha_denominator_bound": enforce_alpha_denominator_bound,
                "score_scale_ceiling": constrained_score_ceiling,
                "score_denominator_floor": (
                    1.0 / constrained_score_ceiling
                    if constrained_score_ceiling is not None
                    else None
                ),
                "min_score_scale": min_score_scale if trainable else None,
                "max_score_scale": max_score_scale if trainable else None,
                "score_denominator": 1.0 / installed_score_scale,
                "score_scale": installed_score_scale,
                "query_scale": query_scale,
                "native_score_scale": native_score_scale,
            }
        )

    return rows


def project_attention_scales_(model: nn.Module, optimizer=None) -> int:
    """Project bounded controllers and clear momentum that points out of bounds.

    Pass the optimizer used for the preceding step so momentum does not leave a
    projected scalar pinned outside the feasible region. The forward
    parameterization independently guarantees the denominator bound.
    """

    seen = set()
    projected_count = 0
    for module in model.modules():
        if (
            not isinstance(module, _ATTENTION_SCALE_CONTROLLER_TYPES)
            or id(module) in seen
        ):
            continue
        seen.add(id(module))
        if not module.project_score_scale_():
            continue
        projected_count += 1
        if optimizer is None:
            continue
        state = optimizer.state.get(module.log_denominator_slack, {})
        for key, value in state.items():
            if key == "step" or not isinstance(value, Tensor):
                continue
            if value.shape == module.log_denominator_slack.shape:
                value.zero_()
    return projected_count


def attention_scale_native_regularization(model: nn.Module) -> Tensor:
    """Mean squared log-distance from each backbone's native score scale."""

    terms = []
    shared_controller = getattr(model, "_qft_learnable_attention_scale", None)
    for module in model.modules():
        installed = getattr(module, "_qft_attention_scaling", None)
        if not isinstance(installed, dict) or not installed.get("trainable"):
            continue
        controller = getattr(module, "_qft_learnable_attention_scale", None)
        if not isinstance(controller, _ATTENTION_SCALE_CONTROLLER_TYPES):
            controller = shared_controller
        if not isinstance(controller, _ATTENTION_SCALE_CONTROLLER_TYPES):
            raise RuntimeError("trainable attention scaling metadata has no controller")
        current_log_scale = controller.score_scale.log()
        native_log_scale = current_log_scale.new_tensor(
            math.log(float(installed["native_score_scale"]))
        )
        terms.append((current_log_scale - native_log_scale).square())
    if not terms:
        raise ValueError("no installed trainable attention scales found")
    return torch.stack(terms).mean()


def set_attention_scales_to_native_(model: nn.Module) -> None:
    """Set every installed controller to its backbone's native score scale."""

    shared_controller = getattr(model, "_qft_learnable_attention_scale", None)
    controller_targets: dict[int, tuple[AttentionScaleController, float]] = {}
    for module in model.modules():
        installed = getattr(module, "_qft_attention_scaling", None)
        if not isinstance(installed, dict) or not installed.get("trainable"):
            continue
        controller = getattr(module, "_qft_learnable_attention_scale", None)
        if not isinstance(controller, _ATTENTION_SCALE_CONTROLLER_TYPES):
            controller = shared_controller
        if not isinstance(controller, _ATTENTION_SCALE_CONTROLLER_TYPES):
            raise RuntimeError("trainable attention scaling metadata has no controller")
        native_scale = float(installed["native_score_scale"])
        existing = controller_targets.get(id(controller))
        if existing is not None and not math.isclose(existing[1], native_scale):
            raise ValueError("one shared controller cannot represent different native scales")
        controller_targets[id(controller)] = (controller, native_scale)
    if not controller_targets:
        raise ValueError("no installed trainable attention scales found")
    # Validate every target before mutating any controller so a mixed model
    # cannot be left partially snapped if one native scale is infeasible.
    for controller, native_scale in controller_targets.values():
        if isinstance(controller, AlphaBoundedAttentionScale):
            controller._validated_log_denominator_slack(native_scale)
        else:
            controller._validated_raw_scale(native_scale)
    for controller, native_scale in controller_targets.values():
        controller.set_score_scale_(native_scale)


def collect_attention_scaling_stats(model: nn.Module) -> list[dict]:
    """Collect current values for every installed learnable score scale."""

    rows = []
    seen = set()

    def append_controller(
        name: str,
        controller: AttentionScaleController,
        shared: bool,
    ) -> None:
        if id(controller) in seen:
            return
        seen.add(id(controller))
        score_scale = float(controller.score_scale.detach().float().cpu().item())
        initial_scale = float(
            controller.initial_score_scale.detach().float().cpu().item()
        )
        ceiling = (
            float(controller.score_scale_ceiling.detach().float().cpu().item())
            if controller.score_scale_ceiling is not None
            else None
        )
        bounded = isinstance(controller, AlphaBoundedAttentionScale)
        row = {
            "name": name,
            "shared": shared,
            "controller": type(controller).__name__,
            "initial_score_scale": initial_scale,
            "score_scale": score_scale,
            "score_denominator": 1.0 / score_scale,
            "effective_alpha_product": 1.0 / math.sqrt(score_scale),
            "log_scale": math.log(score_scale),
            "min_score_scale": float(
                controller.min_score_scale.detach().float().cpu().item()
            ),
            "max_score_scale": float(
                controller.max_score_scale.detach().float().cpu().item()
            ),
            "score_scale_ceiling": ceiling,
            "score_denominator_floor": (
                1.0 / ceiling if ceiling is not None else None
            ),
            "alpha_denominator_bound_enforced": bounded,
            "scale_to_alpha_ratio": (
                score_scale / ceiling if ceiling is not None else None
            ),
            "denominator_to_alpha_ratio": (
                ceiling / score_scale if ceiling is not None else None
            ),
            "constraint_active": (
                bool(controller.log_denominator_slack.detach().cpu().item() <= 0)
                if bounded
                else None
            ),
            "projection_count": controller.projection_count if bounded else 0,
        }
        if bounded:
            row["raw_scale"] = None
            row["log_denominator_slack"] = float(
                controller.log_denominator_slack.detach().float().cpu().item()
            )
        else:
            row["raw_scale"] = float(
                controller.raw_scale.detach().float().cpu().item()
            )
            row["log_denominator_slack"] = None
        rows.append(row)

    shared_controller = getattr(model, "_qft_learnable_attention_scale", None)
    if isinstance(shared_controller, _ATTENTION_SCALE_CONTROLLER_TYPES):
        append_controller("<shared>", shared_controller, True)

    for name, module in model.named_modules():
        controller = getattr(module, "_qft_learnable_attention_scale", None)
        if not isinstance(controller, _ATTENTION_SCALE_CONTROLLER_TYPES):
            continue
        append_controller(name, controller, False)
    return rows
