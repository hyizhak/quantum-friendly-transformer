import pytest
import torch

from alpha_w.measure_weights import canonical_projections, column_statistics, iter_qkv, matrix_norms, projection_descriptor


@pytest.mark.parametrize("prefix,projections", [
    ("bert.encoder.layer.0.attention.self.", ["query", "key", "value"]),
    ("roberta.encoder.layer.0.attention.self.", ["query", "key", "value"]),
    ("distilbert.transformer.layer.0.attention.", ["q_lin", "k_lin", "v_lin"]),
    ("albert.encoder.albert_layer_groups.0.albert_layers.0.attention.", ["query", "key", "value"]),
    ("encoder.layers.0.self_attn.", ["q_proj", "k_proj", "v_proj"]),
    ("encoder.block.0.layer.0.SelfAttention.", ["q", "k", "v"]),
    ("model.layers.0.self_attn.", ["q_proj", "k_proj", "v_proj"]),
])
def test_projection_selection_keeps_q_k_v_distinct(tmp_path, prefix, projections):
    safetensors = pytest.importorskip("safetensors.torch")
    safetensors.save_file({prefix + name + ".weight": torch.eye(3) * i
                          for i, name in enumerate(projections, 1)}, tmp_path / "model.safetensors")
    _, weights, keys = next(iter_qkv(tmp_path, first_only=True))
    assert len(set(keys.values())) == 3
    assert [matrix_norms(weights[k])["spectral"] for k in "qkv"] == [1, 2, 3]


def test_conv1d_and_linear_use_same_paper_column_convention(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    raw = torch.tensor([[3., 4.], [0., 12.]])
    linear = canonical_projections(raw, "q", "linear")["q"]
    packed = torch.cat([raw.T, 2 * raw.T, 3 * raw.T], dim=1)
    conv = canonical_projections(packed, "qkv", "conv1d")
    torch.testing.assert_close(linear, conv["q"])
    assert linear.norm(dim=0).tolist() == [5, 12]
    safetensors.save_file({"h.0.attn.c_attn.weight": packed}, tmp_path / "model.safetensors")
    stats, _ = column_statistics(tmp_path)
    assert stats["Wq_mean"] == 8.5
    assert stats["Wv_mean"] == 25.5


def test_gqa_rectangular_dimensions_and_encoder_only(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    weights = {f"model.layers.0.self_attn.{k}_proj.weight": torch.ones(n, 8)
               for k, n in [("q", 8), ("k", 2), ("v", 2)]}
    safetensors.save_file(weights, tmp_path / "model.safetensors")
    stats, shapes = column_statistics(tmp_path)
    assert shapes[0]["shapes"]["k"] == [8, 2]
    assert stats["d_hidden"] == 8
    assert stats["total_count"] == 12
    assert projection_descriptor("decoder.block.0.layer.0.SelfAttention.q.weight") is None
