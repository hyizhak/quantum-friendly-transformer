import pytest
import torch

from quantum_friendly_transformer.trainer.checkpoints import load_unparametrized_checkpoint
from quantum_friendly_transformer.models import SpectrallyNormalizedTransformerForSequenceClassification


@pytest.mark.parametrize('attention,ffn', [(True, False), (False, True), (True, True)])
def test_vanilla_checkpoint_loads_all_normalized_backbone_weights(attention, ffn):
    args=dict(d_model=8,nhead=2,d_ff=16,num_emb=12,max_seq_len=5,num_classes=2)
    source=SpectrallyNormalizedTransformerForSequenceClassification(**args)
    target=SpectrallyNormalizedTransformerForSequenceClassification(**args,
        apply_attention_sn=attention,apply_ffn_sn=ffn)
    load_unparametrized_checkpoint(target,source.state_dict())
    for name,value in source.state_dict().items():
        key=name if name in target.state_dict() else name.removesuffix('.weight')+'.parametrizations.weight.original'
        torch.testing.assert_close(target.state_dict()[key],value,rtol=0,atol=0)
    incomplete=source.state_dict()
    del incomplete['classifier.weight']
    with pytest.raises(ValueError,match='Missing pretrained'):
        load_unparametrized_checkpoint(target,incomplete)


def test_dnabert_fused_projections_load_into_fn_parameters():
    pytest.importorskip('transformers')
    from quantum_friendly_transformer.norm_transformer.bert.frobenius_normalized_bert import (
        FrobeniuslyNormalizedBertConfig as Config, FrobeniuslyNormalizedBertForSequenceClassification as Model)
    args=dict(vocab_size=20,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16)
    source=Model(Config(**args,apply_attn_fn=False,apply_ffn_fn=False))
    target=Model(Config(**args,apply_attn_fn=True,apply_ffn_fn=True))
    load_unparametrized_checkpoint(target,source.state_dict())
    torch.testing.assert_close(target.bert.embeddings.word_embeddings.weight, source.bert.embeddings.word_embeddings.weight)
    torch.testing.assert_close(target.classifier.weight,source.classifier.weight)
    raw=source.bert.encoder.layer[0].attention.self.Wqkv.weight
    for index,kind in enumerate('qkv'):
        projection=getattr(target.bert.encoder.layer[0].attention.self,'W'+kind)
        torch.testing.assert_close(projection.parametrizations.weight.original,raw.chunk(3)[index])
