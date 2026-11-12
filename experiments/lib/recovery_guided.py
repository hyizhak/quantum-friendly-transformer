"""Training-only distillation and trajectory averages for bounded recovery."""

import math
import re

import torch
from torch.nn import functional as F


def guided_loss(logits, labels, teacher_logits=None, weight=0., temperature=2.):
    ce = F.cross_entropy(logits.float(), labels, reduction='none')
    if not weight:
        return ce.mean(), ce.mean().detach(), ce.new_zeros(())
    assert teacher_logits is not None and 0 <= weight <= 1 and temperature > 0
    teacher = teacher_logits.detach().float()
    eligible = (teacher.argmax(-1) == labels).float()
    kl = F.kl_div(F.log_softmax(logits.float() / temperature, dim=-1),
                  F.softmax(teacher / temperature, dim=-1), reduction='none').sum(-1) * temperature**2
    coefficient = weight * eligible
    # Teacher disagreements leave the gold-label loss fully in control.
    loss = ((1 - coefficient) * ce + coefficient * kl).mean()
    return loss, ce.mean().detach(), (eligible * kl).mean().detach()


def align_training_targets(data, cache):
    assert cache['split'] == 'train'
    ids = cache['example_id'].tolist()
    assert len(set(ids)) == len(ids)
    lookup = {value: index for index, value in enumerate(ids)}
    indices = torch.tensor([lookup[int(value)] for value in data['example_id']], dtype=torch.long)
    assert torch.equal(cache['labels'][indices], data['labels'].cpu())
    logits = cache['logits'][indices]
    assert logits.shape == (len(indices), 2) and bool(torch.isfinite(logits).all())
    return logits


def cyclic_multiplier(step, warmup, cycle, minimum=.25):
    if step <= warmup:
        return step / max(1, warmup)
    phase = ((step - warmup - 1) % cycle) / max(1, cycle - 1)
    return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * phase))


def parameter_groups(model, recipe, weight_decay):
    depth = getattr(model.config, 'num_hidden_layers', None)
    if depth is None:
        depth = len(model.backbone.h)
    groups, scales = {}, []
    for name, p in model.named_parameters():
        assert p.requires_grad
        if '_qft_learnable_attention_scale.' in name:
            scales.append(p)
            continue
        multiplier = 1.
        layer = re.search(r'backbone\.(?:layers|h)\.(\d+)\.', name)
        if layer:
            multiplier = recipe['layer_lr_decay'] ** (depth - 1 - int(layer.group(1)))
        if name == 'score.weight':
            multiplier = recipe['head_lr_multiplier']
        elif 'embed_tokens' in name or '.wte.' in name or '.wpe.' in name:
            multiplier = recipe['embedding_lr_multiplier']
        decay = weight_decay if p.ndim >= 2 else 0.
        key = (multiplier, decay)
        group = groups.setdefault(key, {'params': [], 'lr_multiplier': multiplier,
                                       'lr': recipe['learning_rate'] * multiplier,
                                       'weight_decay': decay, 'eps': 1e-8})
        group['params'].append(p)
    result = list(groups.values())
    ids = [id(p) for group in result for p in group['params']] + [id(p) for p in scales]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    return result, scales


class ParameterAverage:
    """Average learned parameters only; require invariant inference buffers."""

    def __init__(self, model):
        self.names = set(dict(model.named_parameters()))
        self.state = None
        self.count = 0

    @torch.no_grad()
    def update(self, model):
        incoming = model.state_dict()
        if self.state is None:
            self.state = {k: v.detach().cpu().clone() for k, v in incoming.items()}
            self.count = 1
            return
        assert self.state.keys() == incoming.keys()
        self.count += 1
        for key, value in incoming.items():
            value = value.detach().cpu()
            if key in self.names:
                self.state[key].lerp_(value, 1 / self.count)
            else:
                assert torch.equal(self.state[key], value), f'Changed inference buffer: {key}'


def clone_cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
