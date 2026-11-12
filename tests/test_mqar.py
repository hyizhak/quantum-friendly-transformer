import math

import numpy as np
import torch

from experiments.lib.mqar import DataSegment, MQARTransformer, VENDOR, generate_data, validate_data, weight_records
from experiments.lib.mqar_variance_init import MQARVarianceInit


def config():
    return {"vocab_size":128, "dimension":16, "ffn_dimension":32, "layers":2,
            "cap":8., "power_a":.01, "random_non_queries":True}


def test_official_generator_exact_equivalence_and_rng_isolation():
    namespace = {"DataSegmentConfig": object, "DataSegment": DataSegment}
    source = (VENDOR / "multiquery_ar.py").read_text()
    source = source.replace("from zoology.config import DataSegmentConfig", "")
    source = source.replace("from zoology.data.utils import DataSegment", "")
    exec(compile(source, "official_mqar", "exec"), namespace)
    torch.manual_seed(719)
    official = namespace["multiquery_ar"](vocab_size=128, num_examples=8, input_seq_len=32, num_kv_pairs=4, seed=719)
    torch_before, numpy_before = torch.random.get_rng_state(), np.random.get_state()
    actual = generate_data(config(), n=32, pairs=4, examples=8, seed=719)
    assert torch.equal(actual.inputs, official.inputs)
    assert torch.equal(actual.labels, official.labels)
    assert torch.equal(torch_before, torch.random.get_rng_state())
    assert np.array_equal(numpy_before[1], np.random.get_state()[1])
    validate_data(actual, 4, 128)


def test_causality_dense_attention_and_uniform_control():
    torch.manual_seed(17)
    model = MQARTransformer(config()).double().eval()
    inputs = torch.randint(128, (2, 16))
    with torch.inference_mode():
        predicted = model(inputs)
        assert torch.allclose(predicted, model(inputs, dense=True), atol=1e-10, rtol=1e-10)
        changed = inputs.clone()
        changed[:, 9:] = torch.randint(128, changed[:, 9:].shape)
        assert torch.allclose(predicted[:, :9], model(changed)[:, :9], atol=1e-10, rtol=1e-10)
        uniform = model(inputs, force_uniform=True)
        for block in model.blocks:
            block.q.parametrizations.weight.original.zero_()
            block.k.parametrizations.weight.original.zero_()
        assert torch.allclose(uniform, model(inputs, dense=True), atol=1e-10, rtol=1e-10)


def test_norm_constraints_and_identical_variant_initialization():
    torch.manual_seed(19)
    native = MQARTransformer(config())
    torch.manual_seed(19)
    uniform = MQARTransformer(config(), "uniform")
    assert all(torch.equal(a, b) for a,b in zip(native.state_dict().values(), uniform.state_dict().values()))
    with torch.no_grad():
        for block in native.blocks:
            block.q.parametrizations.weight.original.mul_(100)
        inputs = torch.randint(128, (2, 16))
        _, stats = native(inputs, diagnostics=True)
    assert float(native.embed(inputs).detach().double().norm(dim=-1).max()) <= 1
    assert all(abs(s["output_row_norm_max"] - 1) < 1e-6 for s in stats)
    weights = [r for r in weight_records(native) if r["capped"]]
    assert len(weights) == 12
    assert all(r["effective_frobenius"] <= 8 for r in weights)


def test_variance_initialization_matches_ordinary_layernorm_reference():
    cfg = {**config(), "cap": 1000.}
    torch.manual_seed(51)
    reference = MQARTransformer(cfg).double().eval()
    torch.manual_seed(51)
    model = MQARVarianceInit(cfg).double().eval()
    tokens = torch.randint(128, (2, 16))
    with torch.inference_mode():
        hidden = reference.embed(tokens) * math.sqrt(cfg["dimension"])
        for block in reference.blocks:
            q, k, v = block.q(hidden), block.k(hidden), block.v(hidden)
            attention = torch.nn.functional.scaled_dot_product_attention(q[:,None],k[:,None],v[:,None],is_causal=True).squeeze(1)
            hidden = torch.nn.functional.layer_norm(hidden + block.o(attention), (cfg["dimension"],), eps=0)
            hidden = torch.nn.functional.layer_norm(hidden + block.m2(torch.nn.functional.gelu(block.m1(hidden))), (cfg["dimension"],), eps=0)
        assert torch.allclose(model(tokens), reference.readout(hidden), atol=1e-10, rtol=1e-10)
