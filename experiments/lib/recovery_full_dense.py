"""Materialize a bounded adapter checkpoint and train every dense parameter."""

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import parametrize

from experiments.lib.bounded_low_rank import BoundedLowRankWeight
from experiments.lib.recovery_extension import validate_scale_records
from quantum_friendly_transformer.norm_transformer import collect_attention_scaling_stats
from quantum_friendly_transformer.norm_transformer.frobenius_cap import (
    LearnableFrobeniusCap, target_weight_modules,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_verified_parent(model, config):
    reference = config['parent_checkpoint']
    if file_sha256(reference['path']) != reference['sha256']:
        raise ValueError('Parent checkpoint checksum mismatch')
    checkpoint = torch.load(reference['path'], map_location='cpu', weights_only=True, mmap=True)
    assert checkpoint['mode'] == 'bounded' and checkpoint['model'] == 'mistral7b'
    assert checkpoint['config']['cap'] == config['cap']
    for key in ['model_id', 'revision', 'alpha_s', 'baseline_target']:
        assert checkpoint['config']['models']['mistral7b'][key] == config['models']['mistral7b'][key], key
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    validate_scale_records(collect_attention_scaling_stats(model),
                           config['models']['mistral7b']['alpha_s'], config['cap'])


def bf16_embedding_output(_module, _args, output):
    # Preserve the parent's BF16 residual stream while retaining FP32 master
    # embeddings. The cast remains in the autograd graph.
    return output.to(torch.bfloat16)


@torch.no_grad()
def materialize_dense(model, device):
    """Copy each exact rounded effective parent matrix into a dense FP32 weight."""
    rows = []
    for name, module in list(target_weight_modules(model)):
        module.to(device)
        old = module.parametrizations.weight[0]
        assert type(old) is BoundedLowRankWeight
        effective = module.weight.detach().float().cpu()
        gamma_parameter = old.g.detach().cpu().clone()
        cap = LearnableFrobeniusCap(old.max_gamma, eps=old.eps, norm_dtype='float32')
        cap.g.copy_(gamma_parameter)
        rows.append({'layer': name, 'parent_effective_frobenius': float(effective.norm()),
                     'gamma': float(cap.gamma), 'elements': effective.numel()})
        assert float(effective.norm()) < float(cap.gamma)
        parametrize.remove_parametrizations(module, 'weight', leave_parametrized=False)
        module.weight = nn.Parameter(effective)
        parametrize.register_parametrization(module, 'weight', cap)
    model.float()
    model.requires_grad_(True)
    assert not any('lora_' in name for name, _ in model.named_parameters())
    assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())
    model.backbone.embed_tokens.register_forward_hook(bf16_embedding_output)
    return rows


def move_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_tree(v, device) for v in value)
    if isinstance(value, list):
        return [move_tree(v, device) for v in value]
    if isinstance(value, dict):
        return {k: move_tree(v, device) for k, v in value.items()}
    return value


def input_device_hook(device):
    def move(_module, args, kwargs):
        return move_tree(args, device), move_tree(kwargs, device)
    return move


def place_model(model, devices):
    """Sequential model parallelism with autograd-preserving boundary copies."""
    devices = [torch.device(d) for d in devices]
    assert len(devices) in (1, 2)
    backbone = model.backbone
    backbone.embed_tokens.to(devices[0])
    backbone.rotary_emb.to(devices[0])
    model.score.to(devices[0])
    mapping = {'backbone.embed_tokens': str(devices[0]), 'score': str(devices[0])}
    midpoint = (len(backbone.layers) + 1) // 2
    for index, layer in enumerate(backbone.layers):
        device = devices[0] if index < midpoint else devices[-1]
        layer.to(device)
        layer.register_forward_pre_hook(input_device_hook(device), with_kwargs=True)
        mapping[f'backbone.layers.{index}'] = str(device)
    # Returning to GPU 0 also keeps the original classifier's pooling indices
    # and hidden states together. Only the small residual activation crosses.
    backbone.norm.to(devices[0])
    backbone.norm.register_forward_pre_hook(input_device_hook(devices[0]), with_kwargs=True)
    mapping['backbone.norm'] = str(devices[0])
    return mapping


def parameter_inventory(model):
    rows = [{'name': name, 'elements': p.numel(), 'shape': list(p.shape),
             'dtype': str(p.dtype), 'device': str(p.device), 'trainable': p.requires_grad}
            for name, p in model.named_parameters()]
    assert all(row['trainable'] for row in rows)
    return {'total_parameters': sum(r['elements'] for r in rows),
            'trainable_parameters': sum(r['elements'] for r in rows if r['trainable']),
            'parameters': rows}


def probe_indices(elements, count, device):
    count = min(elements, count)
    # Integer arithmetic is essential above 2**24: float32 linspace can round
    # the final index from elements-1 up to elements for large dense matrices.
    return torch.arange(count, device=device, dtype=torch.long) * (elements - 1) // max(1, count - 1)


def parameter_probes(model, count=32):
    return {name: p.detach().reshape(-1)[probe_indices(p.numel(), count, p.device)].clone()
            for name, p in model.named_parameters()}


def gradient_audit(model, before=None):
    now = parameter_probes(model) if before is not None else {}
    rows = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        row = {'name': name, 'gradient_present': grad is not None,
               'gradient_finite': bool(torch.isfinite(grad).all()) if grad is not None else False,
               'gradient_norm': float(grad.norm()) if grad is not None else None}
        if before is not None:
            row['probe_maximum_update'] = float((now[name] - before[name]).abs().max())
        rows.append(row)
    assert all(row['gradient_present'] and row['gradient_finite'] for row in rows)
    return rows
