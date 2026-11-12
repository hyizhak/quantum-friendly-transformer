import torch
from torch import nn

from experiments.lib.mqar import weight_records
from experiments.lib.mqar_standard import MQARStandardTransformer
from experiments.lib.mqar_variance_init import MQARVarianceInit


def config():
    return {"vocab_size":128,"dimension":16,"ffn_dimension":32,"layers":2,
            "cap":None,"layernorm_epsilon":1e-5}


def test_standard_baseline_has_uncapped_weights_and_trainable_layernorm():
    model=MQARStandardTransformer(config())
    assert not any("parametrizations" in name for name,_ in model.named_parameters())
    assert not any(r["capped"] for r in weight_records(model))
    for block in model.blocks:
        for ln in (block.ln1,block.ln2):
            assert isinstance(ln,nn.LayerNorm)
            assert ln.weight.requires_grad and ln.bias.requires_grad
            assert torch.equal(ln.weight,torch.ones_like(ln.weight))
            assert ln.eps==1e-5


def test_initial_function_matches_unit_norm_reparameterization_without_epsilon():
    cfg={**config(),"layernorm_epsilon":0.}
    torch.manual_seed(11)
    standard=MQARStandardTransformer(cfg).double().eval()
    torch.manual_seed(11)
    capped=MQARVarianceInit({**cfg,"cap":64.}).double().eval()
    tokens=torch.randint(128,(2,16))
    with torch.inference_mode():
        assert torch.allclose(standard(tokens),capped(tokens),atol=1e-10,rtol=1e-10)


def test_standard_model_is_causal_and_dense_attention_matches():
    torch.manual_seed(22)
    model=MQARStandardTransformer(config()).double().eval()
    inputs=torch.randint(128,(2,16))
    with torch.inference_mode():
        expected=model(inputs)
        assert torch.allclose(expected,model(inputs,dense=True),atol=1e-10,rtol=1e-10)
        changed=inputs.clone()
        changed[:,9:]=torch.randint(128,changed[:,9:].shape)
        assert torch.allclose(expected[:,:9],model(changed)[:,:9],atol=1e-10,rtol=1e-10)
