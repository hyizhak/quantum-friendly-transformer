"""Regression coverage for sample-to-norm alignment in the legacy measurements."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def measurement():
    pytest.importorskip("datasets")
    pytest.importorskip("transformers")
    path = Path(__file__).resolve().parents[1] / "alpha_s" / "alpha_s.py"
    spec = importlib.util.spec_from_file_location("alpha_s.alpha_s", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tokenizer:
    def __len__(self):
        return 10

    def add_special_tokens(self, tokens):
        pass

    def __call__(self, text):
        return {"input_ids": [int(token) for token in text.split()]}

    def batch_encode_plus(self, texts, **kwargs):
        rows = [self(text)["input_ids"] for text in texts]
        if kwargs.get("truncation"):
            rows = [row[:kwargs["max_length"]] for row in rows]
        width = kwargs["max_length"] if kwargs.get("padding") == "max_length" else max(map(len, rows))
        tokens = torch.tensor([row + [0] * (width - len(row)) for row in rows])
        return {"input_ids": tokens, "attention_mask": tokens.ne(0)}


class Model:
    device = torch.device("cpu")

    def eval(self):
        return self

    def resize_token_embeddings(self, size):
        pass

    def __call__(self, input_ids, **kwargs):
        assert not torch.is_grad_enabled()
        embeddings = torch.nn.functional.one_hot(input_ids % 2, 2).double()
        embeddings *= input_ids.unsqueeze(-1)
        return SimpleNamespace(hidden_states=(embeddings, 2 * embeddings))


@pytest.mark.parametrize("name", ["llama2-7b", "mistral7b", "tinyllama", "gpt2"])
@pytest.mark.parametrize("batch_size", [1, 2, 5])
@pytest.mark.parametrize("max_length", [-1, 2])
def test_saved_norms_match_each_sample(measurement, monkeypatch, tmp_path, name, batch_size, max_length):
    module = measurement
    samples = ["1 2", "3", "4 5 6", "7 8", "9"]
    monkeypatch.setattr(module, "mmlu", lambda: samples)
    monkeypatch.setattr(module, "tqdm", lambda batches: batches)
    monkeypatch.setattr(module, "AutoTokenizer", SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: Tokenizer(),
    ))
    monkeypatch.setattr(module, "AutoModelForCausalLM", SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: Model(),
    ))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_files").mkdir()

    module.main(name, max_length=max_length, batch_size=batch_size, random=False,
                output_dir=tmp_path / "data_files")

    expected = []
    for text in samples:
        tokens = np.array([int(token) for token in text.split()])
        if max_length != -1:
            tokens = tokens[:max_length]
        fro = np.sqrt(np.sum(tokens ** 2))
        # The two columns have disjoint support, so their lengths are the
        # singular values. Each sample deliberately has a different norm.
        spec = max(np.sqrt(np.sum(tokens[tokens % 2 == parity] ** 2)) for parity in (0, 1))
        expected.append([len(tokens), fro, 2 * fro, spec])
    actual = np.load(tmp_path / "data_files" / f"{name}_data.npy")
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("samples, expected", [
    ([], []),
    ([0, 1, 2, 3], [[0, 1], [2, 3]]),
    ([0, 1, 2], [[0, 1], [2]]),
])
def test_batch_inputs_has_no_empty_tail(measurement, samples, expected):
    module = measurement
    assert module.batch_inputs(samples, 2) == expected


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_inputs_requires_positive_size(measurement, batch_size):
    module = measurement
    with pytest.raises(ValueError, match="positive"):
        module.batch_inputs([0, 1], batch_size)


def test_norms_exclude_padding_and_count_the_measured_tokens(measurement):
    embeddings = torch.tensor([[[3., 0.], [0., 4.], [999., 999.]],
                               [[999., 999.], [0., 2.], [0., 0.]]])
    mask = torch.tensor([[1, 1, 0], [0, 1, 0]])
    actual = measurement.measure_batch_norms(embeddings, 2 * embeddings, mask)
    np.testing.assert_allclose(actual, [[2, 5, 10, 4], [1, 2, 4, 2]])


def test_two_decoder_layers_preserve_the_measured_hidden_states():
    transformers = pytest.importorskip("transformers")
    for config_class, model_class in (
        (transformers.LlamaConfig, transformers.LlamaForCausalLM),
        (transformers.MistralConfig, transformers.MistralForCausalLM),
    ):
        options = dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32)
        full = model_class(config_class(num_hidden_layers=3, **options)).eval()
        prefix = model_class(config_class(num_hidden_layers=2, **options)).eval()
        prefix.load_state_dict({key: value for key, value in full.state_dict().items()
                                if key in prefix.state_dict()}, strict=True)
        tokens = torch.tensor([[1, 2, 0], [3, 4, 5]])
        with torch.inference_mode():
            expected = full(tokens, attention_mask=tokens.ne(0), output_hidden_states=True, use_cache=False)
            actual = prefix(tokens, attention_mask=tokens.ne(0), output_hidden_states=True, use_cache=False)
        for layer in (0, 1):
            torch.testing.assert_close(actual.hidden_states[layer], expected.hidden_states[layer], rtol=0, atol=0)


def test_random_inputs_are_reproducible_and_exclude_special_tokens(measurement):
    first = measurement.random_dataset(7, 12, [0, 2, 11], seed=42)
    second = measurement.random_dataset(7, 12, [0, 2, 11], seed=42)
    other = measurement.random_dataset(7, 12, [0, 2, 11], seed=43)
    for i, (a, b) in enumerate(zip(first, second), 1):
        assert a["input_ids"].shape == (1, i)
        assert a["attention_mask"].sum() == i
        assert not set(a["input_ids"].flatten().tolist()) & {0, 2, 11}
        torch.testing.assert_close(a["input_ids"], b["input_ids"])
    assert not torch.equal(first[-1]["input_ids"], other[-1]["input_ids"])
