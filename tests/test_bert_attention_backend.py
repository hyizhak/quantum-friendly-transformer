import pytest
import torch


@pytest.mark.parametrize('normalized', [False, True])
def test_bert_flash_dispatch_uses_same_scale_as_torch(normalized, monkeypatch):
    pytest.importorskip('transformers')
    from quantum_friendly_transformer.norm_transformer.bert import frobenius_normalized_bert as bert
    config = bert.FrobeniuslyNormalizedBertConfig(hidden_size=8, num_attention_heads=2,
        intermediate_size=16, attention_probs_dropout_prob=0, apply_attn_fn=normalized)
    monkeypatch.setattr(bert, 'flash_attn_qkvpacked_func', None)
    attention = bert.FrobeniuslyNormalizedBertUnpadSelfAttention(config).half().eval()
    hidden = torch.randn(5, 8).half()
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    indices = torch.tensor([0, 1, 2, 3, 4])
    lengths = torch.tensor([0, 3, 5], dtype=torch.int32)
    bias = torch.zeros(2, 2, 3, 3).half()
    bias[1, :, :, 2] = -10000
    expected = attention(hidden, lengths, 3, indices, mask, bias)
    scales = []

    def flash(qkv, bias, causal, scale):
        scales.append(scale)
        q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias,
            is_causal=causal, scale=scale).transpose(1, 2)

    monkeypatch.setattr(bert, 'flash_attn_qkvpacked_func', flash)
    actual = attention(hidden, lengths, 3, indices, mask, bias)
    assert scales == [2. if normalized else .5]
    torch.testing.assert_close(actual, expected, rtol=.005, atol=.002)
