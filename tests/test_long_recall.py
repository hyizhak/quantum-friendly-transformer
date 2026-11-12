import pytest
import torch

from experiments.lib.long_recall import (
    LongRecallLayer, VARIANTS, input_certificate, matrix_records, paired_batch,
)


def batch(n=64):
    return paired_batch(4, n, 32, 16, torch.Generator().manual_seed(23))


def test_counterfactual_inputs_have_identical_means_and_opposite_labels():
    rows, labels, selected = batch(1024)
    row_bound, matrix_bound = input_certificate(rows)
    assert row_bound <= 1 and matrix_bound <= 1024
    assert torch.equal(rows[0::2].mean(1), rows[1::2].mean(1))
    assert torch.equal(rows[0::2, -1], rows[1::2, -1])
    assert torch.equal(labels[0::2], 1 - labels[1::2])
    assert torch.equal(selected[0::2], selected[1::2])
    assert torch.equal(rows, batch(1024)[0])


@pytest.mark.parametrize("variant", VARIANTS)
def test_reassociated_attention_matches_direct_matrix_evaluation(variant):
    torch.manual_seed(91)
    model = LongRecallLayer(variant=variant).double()
    rows, _, _ = batch()
    rows = rows.double()
    output, diag = model(rows, diagnostics=True)
    dense, hidden, _ = model.dense_forward(rows)
    torch.testing.assert_close(output, dense, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(diag["hidden"], hidden, atol=1e-12, rtol=1e-12)
    if variant == "strict_frobenius":
        assert diag["scores"].abs().max() <= 1 / rows.shape[1]
    assert len(matrix_records(model)) == 6


def test_uniform_model_stays_at_chance_on_complete_pairs_after_training():
    torch.manual_seed(54)
    model = LongRecallLayer(variant="uniform")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    rows, labels, _ = batch()
    for _ in range(3):
        optimizer.zero_grad()
        output = model(rows)
        torch.nn.functional.cross_entropy(output, labels).backward()
        optimizer.step()
    output = model(rows)
    assert torch.equal(output[0::2], output[1::2])
    assert int((output.argmax(-1) == labels).sum()) == len(labels) // 2


def test_strict_bound_survives_large_raw_weights_and_reload():
    model = LongRecallLayer()
    with torch.no_grad():
        for module in (model.q, model.k, model.v, model.m1, model.m2):
            module.parametrizations.weight.original.normal_(std=50)
    restored = LongRecallLayer()
    restored.load_state_dict(model.state_dict())
    rows, _, _ = batch(1024)
    output, diag = restored(rows, diagnostics=True)
    assert torch.isfinite(output).all()
    assert diag["scores"].abs().max() <= 1 / 1024
    torch.testing.assert_close(output, model(rows))
    assert all(r["effective_frobenius"] <= 24 for r in matrix_records(restored))
