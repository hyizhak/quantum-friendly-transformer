import copy

import pytest
import torch
from transformers import MistralConfig, MistralModel

from experiments.lib.recovery_full_dense import (
    file_sha256, gradient_audit, load_verified_parent, materialize_dense,
    parameter_inventory, parameter_probes, place_model, probe_indices,
)
from experiments.runners.recovery_low_rank import build_model
from experiments.lib.recovery_extension import validate_scale_records
from quantum_friendly_transformer.norm_transformer import collect_attention_scaling_stats, project_attention_scales_
from quantum_friendly_transformer.norm_transformer.frobenius_cap import target_weight_modules


def test_probe_indices_do_not_round_outside_large_matrices():
    for elements in [1, 17, 4096 * 14336, 32000 * 4096]:
        indices = probe_indices(elements, 32, 'cpu')
        assert int(indices[0]) == 0
        assert int(indices[-1]) == elements - 1
        assert bool(((indices >= 0) & (indices < elements)).all())


@pytest.mark.parametrize('devices', [['cpu'], ['cuda:0', 'cuda:1']])
def test_dense_conversion_and_all_parameter_training(tmp_path, devices):
    if devices[0].startswith('cuda') and torch.cuda.device_count() < 2:
        pytest.skip('two GPUs required for model-parallel gradient coverage')
    torch.set_num_threads(2)
    hf = MistralConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                       max_position_embeddings=32, attention_dropout=0.)
    MistralModel(hf).save_pretrained(tmp_path / 'mistral7b')
    recipe = {'cap': .2, 'models': {'mistral7b': {'alpha_s': 5., 'master_dtype': 'bfloat16',
              'train_scope': 'head_caps', 'low_rank_rank': 3, 'low_rank_alpha': 6.,
              'model_id': 'test/mistral', 'revision': 'local', 'baseline_target': .9}}}
    model, _, _ = build_model(tmp_path, 'mistral7b', recipe, 101, 'bounded', torch.device('cpu'))
    # Exercise nonzero adapters instead of only the initialization identity.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'lora_B' in name:
                p.normal_(std=.03)
    parent = tmp_path / 'parent.pth'
    torch.save({'mode': 'bounded', 'model': 'mistral7b', 'config': copy.deepcopy(recipe),
                'model_state_dict': model.state_dict()}, parent)
    recipe['parent_checkpoint'] = {'path': str(parent), 'sha256': file_sha256(parent)}
    load_verified_parent(model, recipe)
    invalid = copy.deepcopy(recipe)
    invalid['models']['mistral7b']['alpha_s'] = 4.
    with pytest.raises(AssertionError):
        load_verified_parent(model, invalid)
    device = torch.device(devices[0])
    model.to(device)
    batch = {'input_ids': torch.randint(0, 64, (4, 8), device=device),
             'attention_mask': torch.ones(4, 8, dtype=torch.long, device=device),
             'labels': torch.tensor([0, 1, 0, 1], device=device)}
    model.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        expected = model(**batch).logits.clone()
    model.cpu()
    conversion = materialize_dense(model, device)
    place_model(model, devices)
    assert len(conversion) == 14
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        torch.testing.assert_close(model(**batch).logits, expected, rtol=0, atol=0)
    inventory = parameter_inventory(model)
    assert inventory['total_parameters'] == inventory['trainable_parameters']
    assert not any('lora_' in name for name, _ in model.named_parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, foreach=False)
    probes = parameter_probes(model)
    model.train()
    with torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=False):
        model(**batch).loss.backward()
    optimizer.step()
    project_attention_scales_(model, optimizer)
    audit = gradient_audit(model, probes)
    for row in audit:
        if '.original' in row['name'] or 'score.weight' == row['name']:
            assert row['probe_maximum_update'] > 0
    assert any(row['probe_maximum_update'] > 0 for row in audit if 'embed_tokens' in row['name'])
    validate_scale_records(collect_attention_scaling_stats(model), 5., .2)
    assert all(float(module.weight.detach().norm()) <= .2 for _, module in target_weight_modules(model))
    checkpoint = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}
    model.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        trained = model(**batch).logits.clone()
    from experiments.runners.recovery_full_dense import build_model as dense_builder
    dense_recipe = copy.deepcopy(recipe)
    dense_recipe['models']['mistral7b'].update(master_dtype='float32', train_scope='all')
    restored, _, _, _, _ = dense_builder(tmp_path, dense_recipe, 202, devices, initialize_parent=False)
    restored.load_state_dict(checkpoint, strict=True)
    restored.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        torch.testing.assert_close(restored(**batch).logits, trained, rtol=0, atol=0)
