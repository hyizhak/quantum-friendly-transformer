import torch

from experiments.lib.recovery_classifier_fit import fit_classifier


def test_full_batch_fit_improves_training_objective_and_preserves_head_center():
    torch.manual_seed(19)
    x = torch.randn(120, 5, dtype=torch.float64)
    y = (x[:, 0] - .4 * x[:, 1] > 0).long()
    teacher = torch.stack([1. - 2*y, 2*y - 1.], -1).double() * 2
    parent = torch.randn(2, 5, dtype=torch.float32) * .01
    fitted, report = fit_classifier(x, y, teacher, parent, torch.arange(120) < 50,
                                    root_fraction=.5, teacher_weight=.5, trust_strength=.1, max_iter=50)
    assert report['final_objective'] < report['initial_objective']
    torch.testing.assert_close(fitted.mean(0), parent.mean(0), rtol=0, atol=1e-7)
    assert float(((x @ fitted.double().T).argmax(-1) == y).double().mean()) > .9
    assert not fitted.requires_grad
