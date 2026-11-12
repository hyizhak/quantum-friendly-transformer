import hashlib

import pytest
import torch
from transformers import OlmoConfig, OlmoModel

from experiments.runners.reviewer_recovery_large import build_model
from experiments.runners.reviewer_recovery_average import apply_average


def test_average_parameters_preserves_bounds_and_rejects_different_buffers(tmp_path):
    cfg = OlmoConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                     num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=16)
    OlmoModel(cfg).save_pretrained(tmp_path / 'olmo1b')
    config = {'cap': .2, 'models': {'olmo1b': {'model_id': 'test/olmo', 'revision': 'local',
              'alpha_s': 5., 'baseline_target': .9, 'master_dtype': 'float32', 'train_scope': 'all'}}}
    parents = []
    references = []
    for seed in [101, 202]:
        model, _, _ = build_model(tmp_path, 'olmo1b', config, seed, 'bounded', torch.device('cpu'))
        state = {k: v.clone() for k, v in model.state_dict().items()}
        parents.append(state)
        path = tmp_path / f'{seed}.pth'
        torch.save({'config': config, 'model': 'olmo1b', 'mode': 'bounded', 'model_state_dict': state}, path)
        references.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    config.update(parent_checkpoints=references, candidate_coefficients={'1': [.5, .5]})
    apply_average(model, config, 'olmo1b', 1)
    for key, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, (parents[0][key] + parents[1][key]) / 2, rtol=0, atol=0)
    path = tmp_path / '202.pth'
    payload = torch.load(path, weights_only=True)
    buffers = set(payload['model_state_dict']) - set(dict(model.named_parameters()))
    assert buffers
    key = next(iter(buffers))
    payload['model_state_dict'][key] = payload['model_state_dict'][key] + 1
    torch.save(payload, path)
    references[1]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(AssertionError):
        apply_average(model, config, 'olmo1b', 1)
