"""Numerical regressions found while auditing the norm measurements and models."""

import pytest
import torch
from torch import nn

from quantum_friendly_transformer.norm_transformer.attention_separate_qkv import AttentionWithSeparateQKV
from quantum_friendly_transformer.norm_transformer.rope import RotaryPositionalEmbedding
from quantum_friendly_transformer.norm_transformer.frobenius_norm import frobenius_norm, frobenius_norm_with_scaling
from quantum_friendly_transformer.norm_transformer.spectral_normalized_transformer import SpectrallyNormalizedTransformerBlock
from quantum_friendly_transformer.norm_transformer.frobenius_normalized_transformer import FrobeniuslyNormalizedTransformerBlock


@pytest.mark.parametrize("head_dim", [4, 8, 16])
def test_rope_preserves_norm_and_relative_position(head_dim):
    rope = RotaryPositionalEmbedding(head_dim, 12)
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(2, 12, 3, head_dim, generator=generator)
    rotated = rope(x)
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1))
    q, k = x[:, :1].expand_as(x), x[:, 1:2].expand_as(x)
    qr, kr = rope(q), rope(k)
    dots = (qr[:, :-3] * kr[:, 3:]).sum(-1)
    torch.testing.assert_close(dots, dots[:, :1].expand_as(dots))


def test_rope_rebuilds_legacy_cache_and_preserves_dtype():
    rope = RotaryPositionalEmbedding(8, 5)
    expected = rope(torch.ones(1, 5, 1, 8))
    state = {"cos": torch.zeros_like(rope.cos), "sin": torch.zeros_like(rope.sin)}
    rope.load_state_dict(state)
    torch.testing.assert_close(rope(torch.ones(1, 5, 1, 8)), expected)
    assert rope(torch.ones(1, 5, 1, 8, dtype=torch.bfloat16)).dtype == torch.bfloat16


@pytest.mark.parametrize("batch_first,unbatched", [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize("bias,extras", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("with_rope", [False, True])
@pytest.mark.parametrize("need_weights", [False, True])
def test_attention_matches_pytorch_layout_masks_and_projections(batch_first, unbatched, bias, extras, with_rope, need_weights):
    torch.manual_seed(6)
    options = dict(embed_dim=8, num_heads=2, dropout=0, bias=bias, batch_first=batch_first,
                   add_bias_kv=extras, add_zero_attn=extras, kdim=6, vdim=4, dtype=torch.float64)
    reference = nn.MultiheadAttention(**options).eval()
    actual = AttentionWithSeparateQKV(**options, rope=nn.Identity() if with_rope else None).eval()
    with torch.no_grad():
        for index, kind in enumerate("qkv"):
            projection = getattr(actual, kind + "_linear")
            projection.weight.copy_(getattr(reference, kind + "_proj_weight"))
            if bias:
                projection.bias.copy_(reference.in_proj_bias.chunk(3)[index])
        actual.out_proj.load_state_dict(reference.out_proj.state_dict())
        if extras:
            actual.bias_k.copy_(reference.bias_k)
            actual.bias_v.copy_(reference.bias_v)
    q, k, v = [torch.randn(2, length, width, dtype=torch.float64)
               for length, width in [(3, 8), (5, 6), (5, 4)]]
    padding = torch.tensor([[False, False, False, False, True], [False, False, False, True, True]])
    mask = torch.zeros(3, 5, dtype=torch.bool)
    mask[:, 1] = True
    if unbatched:
        q, k, v, padding = q[0], k[0], v[0], padding[0]
    elif not batch_first:
        q, k, v = [x.transpose(0, 1) for x in (q, k, v)]
    kwargs = dict(key_padding_mask=padding, attn_mask=mask, need_weights=need_weights, average_attn_weights=False)
    expected, expected_weights = reference(q, k, v, **kwargs)
    output, weights = actual(q, k, v, **kwargs)
    torch.testing.assert_close(output, expected)
    if need_weights:
        torch.testing.assert_close(weights, expected_weights)
    else:
        assert weights is None
    output.square().sum().backward()
    assert actual.out_proj.weight.grad.abs().sum() > 0


def test_rotary_attention_does_not_mix_batches_and_combines_causal_padding():
    attention = AttentionWithSeparateQKV(8, 2, dropout=0, batch_first=True,
                                         rope=RotaryPositionalEmbedding(4, 5)).eval()
    x = torch.randn(2, 5, 8)
    padding = torch.tensor([[False, False, False, False, True], [False, False, False, True, True]])
    output, weights = attention(x, x, x, key_padding_mask=padding, is_causal=True)
    for i in range(2):
        singleton, _ = attention(x[i:i+1], x[i:i+1], x[i:i+1], key_padding_mask=padding[i:i+1], is_causal=True)
        torch.testing.assert_close(output[i:i+1], singleton)
    assert weights.triu(1).count_nonzero() == 0
    assert weights.masked_select(padding[:, None].expand_as(weights)).count_nonzero() == 0


@pytest.mark.parametrize("normalize", [frobenius_norm, frobenius_norm_with_scaling])
def test_half_frobenius_normalization_handles_zero_and_large_weights(normalize):
    layer = nn.Linear(8, 8, bias=False, dtype=torch.float16)
    with torch.no_grad():
        layer.weight.fill_(60000)
    normalize(layer)
    assert torch.isfinite(layer.weight).all()
    assert .9 < layer.weight.float().norm() < 2.01
    layer.weight.float().sum().backward()
    assert torch.isfinite(layer.parametrizations.weight.original.grad).all()
    with torch.no_grad():
        layer.parametrizations.weight.original.zero_()
    assert layer.weight.count_nonzero() == 0


def test_embedding_normalization_preserves_zero_padding():
    spectral = SpectrallyNormalizedTransformerBlock(8, 2, 16, 10, 5, apply_embedding_sn=True)
    assert torch.isfinite(spectral.embedding.weight).all()
    assert spectral.embedding.weight[0].count_nonzero() == 0
    torch.testing.assert_close(spectral.embedding.weight[1:].norm(dim=1), torch.ones(9))
    embedding = nn.Embedding(10, 8)
    with torch.no_grad():
        embedding.weight.zero_()
    frobenius = FrobeniuslyNormalizedTransformerBlock(8, 2, 16, 10, 5,
        apply_embedding_fn=True, embedding_layer=embedding)
    assert frobenius.embedding.weight.count_nonzero() == 0


def test_native_half_cap_zero_has_finite_gradients():
    from quantum_friendly_transformer.norm_transformer.frobenius_cap import LearnableFrobeniusCap
    cap = LearnableFrobeniusCap(2, norm_dtype="native")
    weight = torch.zeros(2, 2, dtype=torch.float16, requires_grad=True)
    output = cap(weight)
    output.sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight))
    assert torch.isfinite(cap.g.grad)
