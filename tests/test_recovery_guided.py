import copy

import pytest
import torch
from torch import nn

from experiments.lib.recovery_guided import (
    ParameterAverage, align_training_targets, cyclic_multiplier, guided_loss, parameter_groups,
)


def test_distillation_uses_gold_for_teacher_errors_and_detaches_teacher():
    logits = torch.tensor([[.5, -.2], [-.1, .3]], requires_grad=True)
    labels = torch.tensor([0, 1])
    wrong_teacher = torch.tensor([[-2., 2.], [3., -3.]], requires_grad=True)
    loss, _, _ = guided_loss(logits, labels, wrong_teacher, .75, 2.)
    torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits, labels), rtol=0, atol=0)
    teacher = torch.tensor([[3., -3.], [-2., 2.]], requires_grad=True)
    loss, _, kd = guided_loss(logits, labels, teacher, .5, 2.)
    loss.backward()
    assert teacher.grad is None and kd > 0
    assert logits.grad[0, 0] < 0 and logits.grad[1, 1] < 0
    assert bool(torch.isfinite(logits.grad).all())


def test_teacher_alignment_checks_ids_labels_and_training_split():
    cache = {'split': 'train', 'example_id': torch.tensor([4, 8, 2]),
             'labels': torch.tensor([0, 1, 0]), 'logits': torch.tensor([[1., 2.], [3., 4.], [5., 6.]])}
    data = {'example_id': torch.tensor([2, 4]), 'labels': torch.tensor([0, 0])}
    assert torch.equal(align_training_targets(data, cache), cache['logits'][[2, 0]])
    wrong = copy.deepcopy(cache)
    wrong['split'] = 'validation'
    with pytest.raises(AssertionError):
        align_training_targets(data, wrong)
    wrong = copy.deepcopy(data)
    wrong['labels'][0] = 1
    with pytest.raises(AssertionError):
        align_training_targets(wrong, cache)


def test_average_learns_parameters_but_preserves_inference_buffers():
    model = nn.Linear(3, 2, bias=False)
    model.register_buffer('floor', torch.tensor(24.))
    average = ParameterAverage(model)
    with torch.no_grad():
        model.weight.fill_(1.)
    average.update(model)
    with torch.no_grad():
        model.weight.fill_(3.)
    average.update(model)
    assert torch.equal(average.state['weight'], torch.full_like(model.weight, 2.))
    assert average.state['floor'] == 24.
    assert torch.equal(model.weight, torch.full_like(model.weight, 3.))
    model.floor.fill_(25.)
    with pytest.raises(AssertionError, match='Changed inference buffer'):
        average.update(model)


def test_learning_rate_groups_cover_a_capped_model_and_prioritize_head(tmp_path):
    from transformers import MistralConfig, MistralModel
    from experiments.runners.reviewer_recovery_large import build_model
    hf = MistralConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
    MistralModel(hf).save_pretrained(tmp_path / 'mistral7b')
    recipe = {'alpha_s': 5., 'master_dtype': 'float32', 'train_scope': 'all',
              'learning_rate': .001, 'layer_lr_decay': .5,
              'embedding_lr_multiplier': .1, 'head_lr_multiplier': 10.}
    model, _, _ = build_model(tmp_path, 'mistral7b', {'cap': .2, 'models': {'mistral7b': recipe}},
                             1, 'bounded', torch.device('cpu'))
    groups, scales = parameter_groups(model, recipe, .01)
    mapping = {id(p): group for group in groups for p in group['params']}
    assert mapping[id(model.score.weight)]['lr'] == .01
    assert mapping[id(model.backbone.embed_tokens.weight)]['lr'] == .0001
    first = model.backbone.layers[0].self_attn.q_proj.parametrizations.weight.original
    last = model.backbone.layers[1].self_attn.q_proj.parametrizations.weight.original
    assert mapping[id(first)]['lr'] == .0005 and mapping[id(last)]['lr'] == .001
    assert len(scales) == 2
    assert cyclic_multiplier(1, 10, 20) == .1
    assert cyclic_multiplier(11, 10, 20) == 1.
    assert cyclic_multiplier(30, 10, 20) == .25
    assert cyclic_multiplier(31, 10, 20) == 1.
