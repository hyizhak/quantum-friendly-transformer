"""Disclosed initialization follow-up preserving unit-norm attention inputs."""

import math

import torch

from experiments.lib.mqar import MQARTransformer


class MQARVarianceInit(MQARTransformer):
    """Initialize projections at the scale of an ordinary post-LN Transformer.

    Relative to ordinary LayerNorm vectors of norm sqrt(d), our unit-norm
    vectors are smaller by sqrt(d). Rescaling Q/K and FFN input weights by
    sqrt(d), FFN output weights by its inverse, and readout by sqrt(d) preserves
    the corresponding initial logits and residual maps when caps are inactive.
    The actual forward-pass caps and native attention denominator are unchanged.
    This is an initialization intervention, not an assertion of identical AdamW
    optimization trajectories under a reparameterization.
    """

    def __init__(self, config, variant="native"):
        super().__init__(config, variant)
        with torch.no_grad():
            scale = math.sqrt(config["dimension"])
            for block in self.blocks:
                for module in (block.q, block.k, block.m1):
                    module.parametrizations.weight.original.mul_(scale)
                block.m2.parametrizations.weight.original.div_(scale)
            self.readout.weight.mul_(scale)
