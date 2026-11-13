"""Load a vanilla checkpoint into an explicitly normalized model."""

import torch
from torch.nn.utils import parametrize


def load_unparametrized_checkpoint(model, state):
    """Map original weights (and DNABERT fused QKV) without ignoring omissions.

    Only newly introduced norm scales and spectral-iteration buffers may retain
    their initialization. Existing embeddings, projections and classifier must
    all be supplied by the source checkpoint.
    """
    target = model.state_dict()
    expanded = {}
    for key, value in state.items():
        if '.Wqkv.' in key and key not in target:
            for kind, part in zip('qkv', value.chunk(3, dim=0)):
                expanded[key.replace('.Wqkv.', '.W' + kind + '.')] = part
        else:
            expanded[key] = value
    converted = {}
    for key, value in expanded.items():
        renamed = key
        if key not in target and key.endswith('.weight'):
            renamed = key[:-len('.weight')] + '.parametrizations.weight.original'
        if renamed not in target:
            raise ValueError(f'Unexpected checkpoint parameter: {key}')
        if value.shape != target[renamed].shape:
            raise ValueError(f'Checkpoint shape mismatch for {key}')
        converted[renamed] = value
    missing = set(target) - set(converted)
    new_parameter_suffixes = (
        '.parametrizations.weight.0.g',
        '.parametrizations.weight.0._u',
        '.parametrizations.weight.0._v',
        '.mlp.residual_scale',
    )
    allowed = {key for key in missing if key.endswith(new_parameter_suffixes)}
    if missing - allowed:
        raise ValueError(f'Missing pretrained parameters: {sorted(missing - allowed)}')
    model.load_state_dict(converted, strict=False)
    # Initialize spectral iteration vectors against the loaded weights, rather
    # than retaining the vectors computed for the random constructor weights.
    with torch.no_grad():
        for module in model.modules():
            if parametrize.is_parametrized(module, 'weight'):
                for transform in module.parametrizations.weight:
                    if hasattr(transform, '_u') and hasattr(transform, '_v'):
                        training = transform.training
                        transform.train()
                        for _ in range(15):
                            transform(module.parametrizations.weight.original)
                        transform.train(training)
    return sorted(allowed)
