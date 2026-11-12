import copy
import hashlib

import pytest
import torch
from transformers import OlmoConfig, OlmoModel

from experiments.runners.reviewer_recovery_large import build_model
from experiments.runners.reviewer_recovery_curriculum import load_parent_state


def test_bounded_parent_restores_outputs_and_rejects_changed_constraint(tmp_path):
    cfg = OlmoConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                     num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=16)
    OlmoModel(cfg).save_pretrained(tmp_path / 'olmo1b')
    config = {'cap': .2, 'models': {'olmo1b': {'model_id': 'test/olmo', 'revision': 'local',
              'alpha_s': 5., 'baseline_target': .9, 'master_dtype': 'float32', 'train_scope': 'all'}}}
    parent, _, _ = build_model(tmp_path, 'olmo1b', config, 101, 'bounded', torch.device('cpu'))
    restored, _, _ = build_model(tmp_path, 'olmo1b', config, 202, 'bounded', torch.device('cpu'))
    path = tmp_path / 'parent.pth'
    torch.save({'config': copy.deepcopy(config), 'mode': 'bounded', 'model': 'olmo1b',
                'model_state_dict': parent.state_dict()}, path)
    config['parent_checkpoint'] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    load_parent_state(restored, config, 'olmo1b')
    batch = {'input_ids': torch.randint(0, 32, (2, 8)), 'attention_mask': torch.ones(2, 8, dtype=torch.long)}
    parent.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(parent(**batch).logits, restored(**batch).logits, rtol=0, atol=0)
    for key, value in [('alpha_s', 4.), ('revision', 'different')]:
        invalid = copy.deepcopy(config)
        invalid['models']['olmo1b'][key] = value
        with pytest.raises(AssertionError):
            load_parent_state(restored, invalid, 'olmo1b')
    invalid = copy.deepcopy(config)
    invalid['cap'] = .3
    with pytest.raises(AssertionError):
        load_parent_state(restored, invalid, 'olmo1b')

    from experiments.runners.reviewer_recovery_head import build_model as head_builder
    probe, _, _ = head_builder(tmp_path, 'olmo1b', config, 151, 'bounded', torch.device('cpu'))
    before = {k: v.clone() for k, v in probe.state_dict().items()}
    assert [n for n, p in probe.named_parameters() if p.requires_grad] == ['score.weight']
    optimizer = torch.optim.AdamW(probe.parameters(), lr=.01)
    probe.train()
    probe(**batch, labels=torch.tensor([0, 1])).loss.backward()
    optimizer.step()
    assert not torch.equal(probe.score.weight, before['score.weight'])
    assert all(torch.equal(value, before[key]) for key, value in probe.state_dict().items() if key != 'score.weight')
    invalid = copy.deepcopy(config)
    invalid['parent_checkpoint']['sha256'] = 'wrong'
    with pytest.raises(AssertionError):
        load_parent_state(restored, invalid, 'olmo1b')
