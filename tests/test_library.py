"""Library contract tests: CPU only, with no dataset or model downloads."""

import importlib
import io
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import quantum_friendly_transformer as qft
from quantum_friendly_transformer import models
from quantum_friendly_transformer.data import LetterTokenizer, build_vocab
from quantum_friendly_transformer.reproducibility import manual_seed
from quantum_friendly_transformer.trainer import legacy_recipes


def test_core_imports_without_optional_frameworks(tmp_path):
    package_root = str(Path(qft.__file__).resolve().parent.parent)
    script = f'''
import importlib.abc
import sys
sys.path.insert(0, {package_root!r})
import torch

blocked = {{"transformers", "einops", "datasets", "sklearn", "pandas",
           "matplotlib", "tqdm", "experiments", "alpha_s", "alpha_w"}}

class NoOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError(fullname, name=fullname)

sys.meta_path.insert(0, NoOptionalImports())
import quantum_friendly_transformer as qft
from quantum_friendly_transformer import data, models, reproducibility, util
from quantum_friendly_transformer.trainer import train, evaluate, trainer_recipe_names
from quantum_friendly_transformer.norm_transformer import bert
import quantum_friendly_transformer.trainer.train_genomic_bench
assert callable(qft.apply_learnable_frobenius_caps)
assert callable(train) and callable(evaluate)
assert trainer_recipe_names()
assert util.LetterTokenizer is data.LetterTokenizer
try:
    bert.FrobeniuslyNormalizedBertConfig
except ImportError as exc:
    assert "[hf]" in str(exc)
else:
    raise AssertionError("BERT should require the hf extra")
'''
    subprocess.run([sys.executable, "-I", "-c", script], cwd=tmp_path, check=True)


def test_frobenius_cap_survives_optimizer_and_checkpoint_reload():
    layer = qft.learnable_frobenius_cap(nn.Linear(8, 8, bias=False), max_gamma=1.0)
    with torch.no_grad():
        layer.parametrizations.weight.original.fill_(3.0)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.1)
    inputs = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    for _ in range(3):
        optimizer.zero_grad()
        layer(inputs).square().mean().backward()
        optimizer.step()
        assert layer.weight.norm().item() <= 1.0
    assert layer.parametrizations.weight[0].g.grad is not None

    checkpoint = io.BytesIO()
    torch.save(layer.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = qft.learnable_frobenius_cap(nn.Linear(8, 8, bias=False), max_gamma=1.0)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored(inputs), layer(inputs))


def test_cap_leaves_small_weights_unchanged():
    layer = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        layer.weight.fill_(0.01)
    original = layer.weight.detach().clone()
    qft.learnable_frobenius_cap(layer, max_gamma=2.0)
    torch.testing.assert_close(layer.weight, original)


@pytest.mark.parametrize(
    ("model_class", "flag_suffix", "token_classifier"),
    [
        (models.FrobeniuslyNormalizedTransformerForSequenceClassification, "fn", False),
        (models.FrobeniuslyNormalizedTransformerForTokenClassification, "fn", True),
        (models.SpectrallyNormalizedTransformerForSequenceClassification, "sn", False),
        (models.SpectrallyNormalizedTransformerForTokenClassification, "sn", True),
    ],
)
def test_reference_models_forward_backward_with_padding(model_class, flag_suffix, token_classifier):
    model = model_class(
        d_model=16, nhead=4, d_ff=32, num_emb=32, max_seq_len=8, num_classes=2,
        **{f"apply_attention_{flag_suffix}": True, f"apply_ffn_{flag_suffix}": True},
    )
    tokens = torch.tensor([[1, 2, 3, 4, 0], [2, 3, 4, 0, 0]])
    output = model(tokens, key_padding_mask=tokens.eq(0))
    expected = (2, 5, 2) if token_classifier else (2, 2)
    assert output.shape == expected
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.classifier.weight.grad is not None


def test_data_and_seeding_preserve_legacy_imports():
    from quantum_friendly_transformer import util

    assert util.LetterTokenizer is LetterTokenizer
    assert util.manual_seed is manual_seed
    tokenizer = LetterTokenizer()
    assert tokenizer("ab\n") == ["<cls>", "<bos>", "a", "b", "<eos>"]
    vocab = build_vocab([("ab", 0), ("bc", 1)], tokenizer, use_padding=True)
    assert {"<pad>", "<unk>", "a", "b", "c"} <= vocab.keys()
    manual_seed(17)
    expected = (random.random(), torch.rand(3))
    manual_seed(17)
    assert random.random() == expected[0]
    torch.testing.assert_close(torch.rand(3), expected[1])


def test_legacy_entrypoints_delegate_without_loading_recipes(monkeypatch):
    calls = []
    monkeypatch.setattr(legacy_recipes, "run_legacy_trainer_recipe", calls.append)
    for recipe in legacy_recipes.LEGACY_TRAINER_RECIPES.values():
        name = f"quantum_friendly_transformer.trainer.{Path(recipe.script).stem}"
        module = importlib.import_module(name)
        # Reload also exercises wrappers already imported elsewhere in the suite.
        importlib.reload(module)
        module.main()
    assert calls == list(legacy_recipes.LEGACY_TRAINER_RECIPES)


def test_installed_recipe_without_checkout_has_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(legacy_recipes, "__file__", str(tmp_path / "package" / "legacy_recipes.py"))
    with pytest.raises(FileNotFoundError, match="requires a source checkout"):
        legacy_recipes.legacy_trainer_script_path("train_genomic_bench")


def test_optional_bert_runs_without_downloads():
    pytest.importorskip("transformers")
    pytest.importorskip("einops")
    from quantum_friendly_transformer.norm_transformer.bert import (
        FrobeniuslyNormalizedBertConfig,
        FrobeniuslyNormalizedBertModel,
    )

    config = FrobeniuslyNormalizedBertConfig(
        vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=4, intermediate_size=32,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        apply_attn_fn=True, apply_ffn_fn=True,
    )
    with pytest.warns(UserWarning, match="defaulting MosaicBERT"):
        model = FrobeniuslyNormalizedBertModel(config).eval()
    tokens = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 0]])
    with torch.no_grad():
        hidden, pooled = model(tokens, attention_mask=tokens.ne(0))
    assert hidden.shape == (2, 4, 16)
    assert pooled.shape == (2, 16)
    assert torch.isfinite(hidden).all()
