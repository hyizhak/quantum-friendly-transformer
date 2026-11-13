from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


@pytest.mark.parametrize("early_stopping", [False, True])
def test_trainer_restores_best_validation_weights(tmp_path, monkeypatch, early_stopping):
    from quantum_friendly_transformer.trainer import util

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([[0., 1.]]))

        def forward(self, x, **kwargs):
            return x.float() @ self.weight

    states = []
    metrics = iter([.1, .9, .8, .9])

    def evaluate(model, *args, **kwargs):
        states.append(model.weight.detach().clone())
        return {"accuracy": next(metrics)}

    monkeypatch.setattr(util, "evaluate", evaluate)
    model = Model()
    loader = [{"input_ids": torch.ones(2, 1, dtype=torch.long),
               "labels": torch.zeros(2, dtype=torch.long), "attention_mask": torch.ones(2, 1)}]
    util.train(model, loader, loader, loader, torch.device("cpu"), str(tmp_path), "model",
               {}, num_epochs=2, lr=.1, is_sequence=False, early_stopping=early_stopping)
    assert not torch.equal(states[1], states[2])
    torch.testing.assert_close(states[-1], states[1], rtol=0, atol=0)
    saved = torch.load(tmp_path / 'model_best.pth', weights_only=True)
    torch.testing.assert_close(model.weight, saved['weight'], rtol=0, atol=0)


@pytest.mark.parametrize("module_name", ["normalisation.glue", "recovery.glue_sequence_classifier"])
def test_eval_loss_is_invariant_to_batch_partition(module_name):
    import importlib
    evaluate = importlib.import_module(
        'quantum_friendly_transformer.experiments.runners.' + module_name).evaluate

    class Model(nn.Module):
        def eval(self):
            return self

        def forward(self, input_ids, labels, **kwargs):
            return SimpleNamespace(loss=input_ids.float().mean(),
                                   logits=torch.nn.functional.one_hot(labels, 2).float())

    data = {"input_ids": torch.tensor([[1], [2], [3], [10]]), "labels": torch.tensor([0, 1, 0, 1])}
    whole = evaluate(torch, Model(), [data], 'cpu', 'sst2')
    split = evaluate(torch, Model(), [{k:v[:3] for k,v in data.items()},
                                     {k:v[3:] for k,v in data.items()}], 'cpu', 'sst2')
    assert split == whole
    assert whole['eval_loss'] == 4


def test_singleton_model_output_tuple_metrics():
    pytest.importorskip('sklearn')
    from quantum_friendly_transformer.trainer.metrics import compute_metrics
    logits = np.array([[2., 0.], [0., 3.]])
    assert compute_metrics(((logits,), np.array([0, 1]))) == {'f1': 1., 'accuracy': 1.}


def test_resource_denominator_includes_second_rms_norm():
    pd = pytest.importorskip('pandas')
    from quantum_friendly_transformer.experiments.lib.aggregation import resource_constants, hidden_growth
    group = {'model':'tiny', 'dataset':'test', 'task':'lm', 'seq_len':8}
    norms = pd.DataFrame([{**group, 'metric':metric, 'p5':value, 'median':value,
                           'p95':value, 'sample_count':10}
                          for metric,value in [('rms_denom',2.),('rms2_denom',.25)]])
    result = resource_constants(pd.DataFrame([group]), None, norms, None)
    assert result.iloc[0]['normalisation_denom_p5'] == .25
    incomplete = pd.DataFrame([{**group, 'layer':0, 'hidden_fro_over_sqrtN':1., 'hidden_p95_row_norm':1.}])
    assert hidden_growth(incomplete).empty


def test_spectral_projection_caps_matrix_with_null_all_ones_direction():
    from quantum_friendly_transformer.experiments.lib.common import project_model_weights
    model = nn.Module()
    model.encoder = nn.Module()
    layer = nn.Module()
    layer.attention = nn.Module()
    setattr(layer.attention, 'self', nn.Module())
    layer.attention.self.query = nn.Linear(2, 2, bias=False)
    model.encoder.layer = nn.ModuleList([layer])
    with torch.no_grad():
        layer.attention.self.query.weight.copy_(torch.tensor([[2., -2.], [-2., 2.]]))
    assert project_model_weights(model, 'sn', 1.) == 1
    assert torch.linalg.matrix_norm(layer.attention.self.query.weight, ord=2) <= 1.000001
