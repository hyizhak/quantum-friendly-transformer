import pytest
import torch

from experiments.lib.recovery_sam import sharpness_perturbation


def test_sam_gradient_at_perturbed_point_and_exact_restoration():
    p = torch.nn.Parameter(torch.tensor([3., 4.]))
    (p.square().sum()/2).backward()
    original = p.detach().clone()
    with sharpness_perturbation([p], .5) as norm:
        assert norm == 5.
        torch.testing.assert_close(p-original, torch.tensor([.3, .4]))
        assert float((p-original).norm()) == pytest.approx(.5)
        p.grad = None
        (p.square().sum()/2).backward()
    assert torch.equal(p, original)
    torch.testing.assert_close(p.grad, torch.tensor([3.3, 4.4]))


def test_sam_restores_after_exception_and_rejects_bad_gradient():
    p = torch.nn.Parameter(torch.tensor([1., -2.]))
    p.grad = torch.tensor([.7, -.3])
    original = p.detach().clone()
    with pytest.raises(RuntimeError, match='intentional'):
        with sharpness_perturbation([p], .1):
            raise RuntimeError('intentional')
    assert torch.equal(p, original)
    p.grad.fill_(float('nan'))
    with pytest.raises(FloatingPointError):
        with sharpness_perturbation([p], .1):
            pass

