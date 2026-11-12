import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

from experiments.lib.bounded_low_rank import BoundedLowRankWeight
from quantum_friendly_transformer.norm_transformer.frobenius_cap import LearnableFrobeniusCap


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_composed_weight_cap_gradients_and_reload(dtype, tmp_path):
    torch.manual_seed(101)
    layer = nn.Linear(8, 6, bias=False, dtype=dtype)
    base = layer.weight.detach().clone()
    layer.weight.requires_grad_(False)
    cap = LearnableFrobeniusCap(.2)
    low_rank = BoundedLowRankWeight(layer.weight, cap, rank=3, alpha=6.)
    parametrize.register_parametrization(layer, 'weight', low_rank)
    optimizer = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            loss = layer(torch.randn(4, 8)).square().sum()
        loss.backward()
        assert low_rank.lora_B.grad.abs().sum() > 0
        optimizer.step()
    assert torch.equal(layer.parametrizations.weight.original, base)
    assert low_rank.lora_A.grad.abs().sum() > 0
    with torch.no_grad():
        low_rank.lora_B.fill_(100.)
        low_rank.g.fill_(30.)
        assert float(layer.weight.float().norm()) <= .2
    checkpoint = tmp_path / 'weights.pth'
    torch.save(layer.state_dict(), checkpoint)
    restored = nn.Linear(8, 6, bias=False, dtype=dtype)
    parametrize.register_parametrization(restored, 'weight', BoundedLowRankWeight(restored.weight, LearnableFrobeniusCap(.2), 3, 6.))
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert torch.equal(restored.weight, layer.weight)


@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_mistral_adapter_preserves_effective_bounds(tmp_path, dtype):
    from transformers import MistralConfig, MistralModel
    from experiments.runners.recovery_low_rank import build_model
    from experiments.lib.recovery_extension import validate_scale_records
    from quantum_friendly_transformer.norm_transformer import collect_attention_scaling_stats, project_attention_scales_
    from quantum_friendly_transformer.norm_transformer.frobenius_cap import target_weight_modules

    config = MistralConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                           num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                           max_position_embeddings=32)
    MistralModel(config).save_pretrained(tmp_path / 'mistral7b')
    recipe = {'cap': .2, 'models': {'mistral7b': {'alpha_s': 5., 'master_dtype': dtype,
              'train_scope': 'head_caps', 'low_rank_rank': 3, 'low_rank_alpha': 6.}}}
    model, count, _ = build_model(tmp_path, 'mistral7b', recipe, 101, 'bounded', torch.device('cpu'))
    assert count == 14
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=.001)
    batch = {'input_ids': torch.randint(0, 64, (2, 8)), 'attention_mask': torch.ones(2, 8, dtype=torch.long),
             'labels': torch.tensor([0, 1])}
    model.train()
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=dtype == 'bfloat16'):
        loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    project_attention_scales_(model, optimizer)
    validate_scale_records(collect_attention_scaling_stats(model), 5., .2)
    for _, module in target_weight_modules(model):
        assert module.parametrizations.weight.original.grad is None
        assert float(module.weight.detach().float().norm()) <= .2
    model.eval()
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=dtype == 'bfloat16'):
        expected = model(**batch).logits
    checkpoint = tmp_path / 'model.pth'
    torch.save(model.state_dict(), checkpoint)
    restored, _, _ = build_model(tmp_path, 'mistral7b', recipe, 202, 'bounded', torch.device('cpu'))
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    restored.eval()
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=dtype == 'bfloat16'):
        torch.testing.assert_close(restored(**batch).logits, expected, rtol=0, atol=0)
