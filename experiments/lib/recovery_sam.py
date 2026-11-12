"""SAM perturbations that restore the original parameters exactly."""

from contextlib import contextmanager

import torch


@contextmanager
def sharpness_perturbation(parameters, radius):
    if radius < 0:
        raise ValueError('The SAM radius must be nonnegative')
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        raise ValueError('SAM requires gradients from an unperturbed loss')
    norm = torch.stack([p.grad.detach().float().norm() for p in parameters]).norm()
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('Non-finite first-pass gradient')
    originals = [p.detach().clone() for p in parameters]
    try:
        with torch.no_grad():
            scale = float(radius / (norm + 1e-12))
            for p in parameters:
                p.add_(p.grad, alpha=scale)
        yield float(norm)
    finally:
        with torch.no_grad():
            for p, original in zip(parameters, originals):
                p.copy_(original)

