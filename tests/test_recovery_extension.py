import math

import pytest
import torch
from torch import nn
from transformers import GPTNeoConfig, GPTNeoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from experiments.lib.recovery_extension import (
    AttentionInputMonitor, install_uniform_attention, partition_training,
    sentence_key, validate_scale_records,
)
from quantum_friendly_transformer.experiments.runners.recovery.glue_sequence_classifier import make_decoder_sequence_classifier
from quantum_friendly_transformer.norm_transformer import (
    apply_attention_scaling, collect_attention_scaling_stats, project_attention_scales_,
)


def test_sentence_groups_and_official_evaluation_are_disjoint():
    records = [{"sentence": s} for s in ["A film", "a  FILM", "good", "bad", "nice", "poor", "official"]]
    official = [{"sentence": "OFFICIAL"}]
    split = partition_training(records, official, .4, 71)
    keys = {name: {sentence_key(records[i]["sentence"]) for i in split[name]} for name in ["train", "selection"]}
    assert not keys["train"] & keys["selection"]
    assert "official" not in keys["train"] | keys["selection"]
    assert split["excluded_evaluation_overlap"] == [6]
    assert sorted(split["train"] + split["selection"]) == list(range(6))
    assert split == partition_training(records, official, .4, 71)


def tiny_model():
    config = GPTNeoConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_layers=2, num_heads=2, max_position_embeddings=32,
                          attention_types=[[["global", "local"], 1]], window_size=4,
                          resid_dropout=0., embed_dropout=0., attention_dropout=0.,
                          use_cache=False, problem_type="single_label_classification")
    config._attn_implementation = "eager"
    return make_decoder_sequence_classifier(torch, nn, SequenceClassifierOutput, GPTNeoModel(config), 2)


def test_real_backbone_bounded_training_checkpoint_round_trip(tmp_path):
    torch.manual_seed(4)
    model = tiny_model()
    apply_attention_scaling(model, "alpha_s_alpha_w", alpha_s=4., alpha_w=24.,
                            trainable=True, enforce_alpha_denominator_bound=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    batch = {"input_ids": torch.randint(0, 32, (2, 8)), "attention_mask": torch.ones(2, 8, dtype=torch.long),
             "labels": torch.tensor([0, 1])}
    for _ in range(3):
        optimizer.zero_grad()
        model(**batch).loss.backward()
        optimizer.step()
        project_attention_scales_(model, optimizer)
        validate_scale_records(collect_attention_scaling_stats(model), 4., 24.)
    model.eval()
    before = model(**batch).logits.detach()
    with AttentionInputMonitor(model, 4.) as monitor:
        after = model(**batch).logits.detach()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    rows = monitor.rows()
    assert len(rows) == 2
    assert all(r["maximum_matrix_frobenius"] > r["maximum_token_l2"] for r in rows)
    assert all(math.isfinite(r["maximum_matrix_to_alpha_ratio"]) for r in rows)
    path = tmp_path / "state.pth"
    torch.save(model.state_dict(), path)
    restored = tiny_model()
    apply_attention_scaling(restored, "alpha_s_alpha_w", alpha_s=4., alpha_w=24.,
                            trainable=True, enforce_alpha_denominator_bound=True)
    restored.load_state_dict(torch.load(path, weights_only=True))
    restored.eval()
    torch.testing.assert_close(restored(**batch).logits, before, rtol=0, atol=0)


def test_uniform_control_uses_existing_causal_and_local_masks():
    model = tiny_model().eval()
    assert install_uniform_attention(model) == 2
    x = torch.randint(0, 32, (1, 8))
    result = model.backbone(x, attention_mask=torch.ones_like(x), output_attentions=True)
    global_row = result.attentions[0][0, 0, -1]
    local_row = result.attentions[1][0, 0, -1]
    torch.testing.assert_close(global_row, torch.full((8,), 1 / 8))
    torch.testing.assert_close(local_row[:4], torch.zeros(4))
    torch.testing.assert_close(local_row[4:], torch.full((4,), 1 / 4))


def test_scale_validation_rejects_a_relaxed_controller():
    with pytest.raises(ValueError, match="escaped"):
        validate_scale_records([{"alpha_denominator_bound_enforced": True,
                                 "score_scale": .5, "score_denominator": 2.}], 4., 24.)


def test_posthoc_attention_uses_final_unmasked_query():
    from experiments.runners.validate_recovery_extension import measure_attention

    model = tiny_model().eval()
    install_uniform_attention(model)
    data = {"input_ids": torch.randint(0, 32, (2, 8)),
            "attention_mask": torch.tensor([[1] * 8, [1, 1, 1, 0, 0, 0, 0, 0]]),
            "labels": torch.tensor([0, 1])}
    rows = measure_attention(model, data, torch.device("cpu"))
    assert len(rows) == 2
    for row in rows:
        assert row["head_query_observations"] == 4
        assert row["visible_keys_times_max_probability_maximum"] == pytest.approx(1., abs=1e-6)
        assert row["normalized_entropy_minimum"] == pytest.approx(1., abs=1e-6)
