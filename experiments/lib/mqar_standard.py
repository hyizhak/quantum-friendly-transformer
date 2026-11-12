"""Matched ordinary Transformer baseline for the MQAR comparison."""

import math

from torch import nn
from torch.nn.utils import parametrize

from experiments.lib.mqar import MQARTransformer


class MQARStandardTransformer(MQARTransformer):
    """Use ordinary trainable LayerNorm and uncapped projection matrices.

    Keep the two-layer causal architecture, width, positional features and
    vocabulary readout. Scale the normalized input features by sqrt(d) for
    ordinary per-coordinate activation variance. The initial random parameter
    draws match the corresponding QFT adaptation before its explicit weight
    rescaling. This is a matched baseline, not Zoology's whole model sweep.
    """

    def __init__(self, config, variant="native"):
        # Reuse the exact parameter draw order, then remove the parametrizations.
        super().__init__({**config, "cap": 64.0}, variant)
        self.config = config
        d = config["dimension"]
        for block in self.blocks:
            for name in ("q", "k", "v", "o", "m1", "m2"):
                parametrize.remove_parametrizations(getattr(block, name), "weight", leave_parametrized=False)
            block.ln1 = nn.LayerNorm(d, eps=config["layernorm_epsilon"])
            block.ln2 = nn.LayerNorm(d, eps=config["layernorm_epsilon"])

    def embed(self, tokens):
        return super().embed(tokens) * math.sqrt(self.config["dimension"])
