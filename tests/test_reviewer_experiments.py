"""Checks for numerical comparisons and experimental controls."""

import math

import numpy as np
import pytest
import torch

from experiments.lib.reviewer import attention_statistics, sampled_attention, softmax, taylor_degree
from quantum_friendly_transformer.models import FrobeniuslyNormalizedTransformerForSequenceClassification


def test_sampler_with_all_rows_equals_exact_attention():
    rng = np.random.default_rng(2)
    rows = rng.standard_normal((17, 5))
    query = rng.standard_normal(5)
    value_weight = rng.standard_normal((5, 3))
    exact = softmax(rows @ query) @ (rows @ value_weight)
    actual = sampled_attention(rows, query, value_weight, np.arange(len(rows)))
    np.testing.assert_allclose(actual, exact, rtol=1e-12, atol=1e-12)


def test_uniform_attention_statistics_and_sampling():
    stats = attention_statistics(np.zeros(32))
    assert stats["normalized_entropy"] == pytest.approx(1)
    assert stats["max_probability_times_keys"] == pytest.approx(1)
    assert stats["participation_ratio"] == pytest.approx(32)
    rows = np.eye(4)
    np.testing.assert_allclose(sampled_attention(rows, np.zeros(4), np.eye(4), np.arange(4)), np.ones(4) / 4)


@pytest.mark.parametrize("n", [2, 256, 65536])
def test_selective_family_scalar_formula(n):
    logits = np.full(n, -math.log(n) / 2)
    logits[0] *= -1
    p = softmax(logits)
    assert p[0] == pytest.approx(n / (2 * n - 1))
    assert 2 * p[0] - 1 == pytest.approx(1 / (2 * n - 1), abs=1e-14)


@pytest.mark.parametrize("kappa", [1, 10, 100])
def test_taylor_degree_meets_uniform_remainder_budget(kappa):
    delta = 1e-7
    degree = taylor_degree(kappa, delta)
    a = kappa / 2
    assert a + (degree + 1) * math.log(a) - math.lgamma(degree + 2) <= math.log(delta)


def test_reference_attention_evaluation_disables_dropout():
    torch.manual_seed(4)
    model = FrobeniuslyNormalizedTransformerForSequenceClassification(
        d_model=16, nhead=4, d_ff=32, num_emb=16, max_seq_len=8, num_classes=2,
    ).eval()
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    with torch.no_grad():
        first, second = model(tokens), model(tokens)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["uniform", "rounding"])
def test_interface_noise_obeys_unit_coordinate_bound(kind):
    from experiments.lib.reviewer_pretrained import perturb_interface

    torch.manual_seed(5)
    x = torch.randn(2, 5, 16) * torch.arange(1, 6).view(1, 5, 1)
    x[0, 0] = 0
    epsilon = 0.1
    y = perturb_interface(x, epsilon, kind, torch.Generator().manual_seed(8))
    norm = x.norm(dim=-1, keepdim=True)
    assert ((y - x).abs() <= epsilon * norm / math.sqrt(16) + 1e-6).all()
    assert torch.equal(x[0, 0], y[0, 0])
    assert torch.equal(perturb_interface(x, 0, kind), x)


def test_real_gptneo_diagnostics_preserve_outputs_and_observe_local_mask():
    pytest.importorskip("transformers")
    from transformers import GPTNeoConfig, GPTNeoForCausalLM
    from experiments.lib.reviewer_pretrained import GPTNeoDiagnostics
    from experiments.runners.reviewer_pretrained import score
    from torch.nn import functional as F

    config = GPTNeoConfig(vocab_size=32, hidden_size=16, num_layers=2, num_heads=4,
                          intermediate_size=32, max_position_embeddings=32,
                          attention_types=[[["global", "local"], 1]], window_size=4,
                          attention_dropout=0, resid_dropout=0, embed_dropout=0)
    config._attn_implementation = "eager"
    model = GPTNeoForCausalLM(config).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    with torch.inference_mode():
        expected = model(tokens, use_cache=False).logits
        diag = GPTNeoDiagnostics(model, "native", 8)
        actual = model(tokens, use_cache=False).logits
        loss, _ = score(model, tokens, 3)
        expected_loss = F.cross_entropy(expected[:, -4:-1].reshape(-1, 32), tokens[:, -3:].reshape(-1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert loss == pytest.approx(float(expected_loss), abs=1e-6)
    assert {r["visible_keys"] for r in diag.attention if r["layer"] == 0} == {8}
    assert {r["visible_keys"] for r in diag.attention if r["layer"] == 1} == {4}
    diag.close()


def test_recall_examples_are_unit_norm_and_labels_are_not_in_query():
    from experiments.lib.reviewer_recall import recall_batch

    rows, labels, selected = recall_batch(8, 32, 16, 8, 4, torch.Generator().manual_seed(3))
    torch.testing.assert_close(rows.norm(dim=-1), torch.ones(8, 32))
    batch = torch.arange(8)
    torch.testing.assert_close(rows[:, -1, :8], rows[batch, selected, :8])
    assert torch.equal(rows[batch, selected, 8:12].argmax(-1), labels)
    assert torch.equal(rows[:, -1, 8:12], torch.zeros(8, 4))


def test_recall_scaling_bound_and_key_permutation_invariance():
    from experiments.lib.reviewer_recall import RecallLayer, matrix_stats, recall_batch

    rows, labels, _ = recall_batch(4, 32, 16, 8, 4, torch.Generator().manual_seed(4))
    model = RecallLayer(16, 32, 4, 24, "theorem_frobenius")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    output, diag = model(rows, diagnostics=True)
    assert float(diag["logits"].detach().abs().max()) <= 1 / 32
    permutation = torch.cat([torch.arange(30, -1, -1), torch.tensor([31])])
    torch.testing.assert_close(model(rows[:, permutation]), output, atol=1e-6, rtol=1e-6)
    torch.nn.functional.cross_entropy(output, labels).backward()
    optimizer.step()
    assert len(matrix_stats(model)) == 6
