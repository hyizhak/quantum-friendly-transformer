"""Low-rank weight adaptation with a cap on the complete effective matrix."""

import math
import torch
from torch import nn

from quantum_friendly_transformer.norm_transformer.frobenius_cap import (
    LearnableFrobeniusCap, target_weight_modules,
)


class BoundedLowRankWeight(LearnableFrobeniusCap):
    def __init__(self, weight, cap, rank, alpha):
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('rank and alpha must be positive')
        super().__init__(cap.max_gamma, eps=cap.eps, norm_dtype='float32')
        self.g = cap.g
        self.rank = rank
        self.low_rank_scale = float(alpha) / rank
        self.lora_A = nn.Parameter(torch.empty(rank, weight.shape[1], device=weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(weight.shape[0], rank, device=weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, weight):
        with torch.autocast(device_type=weight.device.type, enabled=False):
            combined = weight.float() + self.low_rank_scale * (self.lora_B @ self.lora_A)
            norm = torch.linalg.vector_norm(combined)
            # Reserve room for the final cast's rounding error. The complete
            # matrix, including the adapter, stays under the configured cap.
            margin = 1 - 2 * torch.finfo(weight.dtype).eps
            scale = (self.gamma * margin / (norm + self.eps)).clamp(max=1.)
            return (combined * scale).to(weight.dtype)


def install_bounded_low_rank(model, rank, alpha):
    count = 0
    for name, module in target_weight_modules(model):
        if not hasattr(module, 'parametrizations') or 'weight' not in module.parametrizations:
            raise ValueError(f'Expected a capped matrix: {name}')
        parameters = module.parametrizations.weight
        if len(parameters) != 1 or type(parameters[0]) is not LearnableFrobeniusCap:
            raise ValueError(f'Unexpected existing weight parametrization: {name}')
        parameters.original.requires_grad_(False)
        parameters[0] = BoundedLowRankWeight(parameters.original, parameters[0], rank, alpha)
        count += 1
    if not count:
        raise ValueError('No capped weight matrices found')
    return count
