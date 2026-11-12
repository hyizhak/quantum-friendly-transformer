"""Exercise the large-backbone adapters before launching expensive jobs."""

import pytest
import torch
from transformers import MistralConfig, MistralModel, OlmoConfig, OlmoModel

from experiments.runners.recovery_large import LargeAttentionInputMonitor, build_model
from experiments.lib.recovery_extension import validate_scale_records
from quantum_friendly_transformer.norm_transformer import collect_attention_scaling_stats, project_attention_scales_


@pytest.mark.parametrize('kind,scope,dtype', [('olmo', 'all', 'float32'),
                                            ('mistral', 'head_caps', 'float32'),
                                            ('mistral', 'head_caps', 'bfloat16')])
def test_large_adapter_training_and_checkpoint(tmp_path, kind, scope, dtype):
    torch.manual_seed(42)
    cls, model_cls = (OlmoConfig, OlmoModel) if kind == 'olmo' else (MistralConfig, MistralModel)
    cfg = cls(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
              num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
    model_cls(cfg).save_pretrained(tmp_path / kind)
    recipe = {'cap': .2, 'models': {kind: {'alpha_s': 5., 'master_dtype': dtype, 'train_scope': scope}}}
    model, caps, _ = build_model(tmp_path, kind, recipe, 101, 'bounded', torch.device('cpu'))
    assert caps == 14
    if scope == 'head_caps':
        assert all(not p.requires_grad for n, p in model.named_parameters() if n.endswith('parametrizations.weight.original'))
        assert model.score.weight.requires_grad
    batch = {'input_ids': torch.randint(0, 64, (2, 8)), 'attention_mask': torch.ones(2, 8, dtype=torch.long),
             'labels': torch.tensor([0, 1])}
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=dtype == 'bfloat16'):
        loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    project_attention_scales_(model, optimizer)
    validate_scale_records(collect_attention_scaling_stats(model), 5., .2)
    model.eval()
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=dtype == 'bfloat16'):
        expected = model(**batch).logits
        with LargeAttentionInputMonitor(model, 5.) as monitor:
            measured = model(**batch).logits
    assert torch.equal(expected, measured)
    assert len(monitor.rows()) == 2
    path = tmp_path / 'checkpoint.pth'
    torch.save(model.state_dict(), path)
    restored, _, _ = build_model(tmp_path, kind, recipe, 202, 'bounded', torch.device('cpu'))
    restored.load_state_dict(torch.load(path, weights_only=True))
    restored.eval()
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=dtype == 'bfloat16'):
        torch.testing.assert_close(restored(**batch).logits, expected, rtol=0, atol=0)
