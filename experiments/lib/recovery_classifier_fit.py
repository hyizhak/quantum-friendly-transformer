"""Deterministic, bias-free classifier fitting on frozen training features."""

import torch
from torch.nn import functional as F


def fit_classifier(features, labels, teacher_logits, parent_weight, root_mask,
                   root_fraction, teacher_weight, trust_strength, max_iter=100):
    assert features.ndim == 2 and parent_weight.shape == (2, features.shape[1])
    assert 0 <= root_fraction <= 1 and 0 <= teacher_weight <= 1 and trust_strength >= 0
    x = features.detach().double()
    y = labels.detach().double()
    prior = parent_weight.detach().double()
    teacher = teacher_logits.detach().double()
    root_mask = root_mask.double()
    weights = torch.full_like(y, (1 - root_fraction) / len(y))
    weights += root_fraction * root_mask / root_mask.sum()
    assert abs(float(weights.sum()) - 1.) < 1e-10
    initial_direction = prior[1] - prior[0]
    initial_margin = x @ initial_direction
    direction = torch.nn.Parameter(initial_direction.clone())
    temperature = 2.
    soft = torch.softmax(teacher / temperature, -1)[:, 1]
    eligible = (teacher.argmax(-1) == labels).double()
    coefficient = teacher_weight * eligible
    entropy = -(soft * soft.clamp_min(1e-15).log() + (1-soft) * (1-soft).clamp_min(1e-15).log())
    optimizer = torch.optim.LBFGS([direction], max_iter=max_iter, history_size=20,
                                  line_search_fn='strong_wolfe', tolerance_grad=1e-8, tolerance_change=1e-12)
    calls = 0

    def objective():
        margin = x @ direction
        ce = F.binary_cross_entropy_with_logits(margin, y, reduction='none')
        kd = (F.binary_cross_entropy_with_logits(margin / temperature, soft, reduction='none') - entropy) * temperature**2
        anchor = (margin - initial_margin).square()
        return (weights * ((1 - coefficient) * ce + coefficient * kd + trust_strength * anchor)).sum()

    initial_loss = float(objective().detach())

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        assert bool(torch.isfinite(loss))
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        center = prior.mean(0)
        fitted = torch.stack([center - .5 * direction, center + .5 * direction]).to(parent_weight.dtype)
        final_loss = float(objective())
    assert final_loss <= initial_loss + 1e-9
    return fitted, {'initial_objective': initial_loss, 'final_objective': final_loss,
                    'closure_calls': calls, 'solver_iterations': optimizer.state[direction]['n_iter'],
                    'root_fraction': root_fraction, 'teacher_weight': teacher_weight,
                    'functional_trust_strength': trust_strength}
