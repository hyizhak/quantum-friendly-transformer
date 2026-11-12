import math

import numpy as np
import pytest
import torch

from experiments.lib.selective_layer import (
    SelectiveLayer, grover_parameters, input_rows, simulate_search_state,
)


@pytest.mark.parametrize('n', [16, 63])
@pytest.mark.parametrize('eps', [0., 1e-5])
def test_complete_layer_selectivity_conditioning_and_position_invariance(n, eps):
    model = SelectiveLayer(n, eps)
    absent = input_rows(n)
    out0, diag0 = model(absent)
    assert float(diag0['attention_average']) == pytest.approx(-1.)
    for position in [0, n-2]:
        present = input_rows(n, position)
        assert torch.equal(absent.abs(), present.abs())
        torch.testing.assert_close(present.norm(dim=-1), torch.ones(n, dtype=torch.float64))
        out1, diag1 = model(present)
        assert float(diag1['probabilities'][position]) == pytest.approx(n/(2*n-1))
        assert float(diag1['attention_average']) == pytest.approx(1/(2*n-1), abs=1e-14)
        assert float((out1-out0).norm()) > .69
        assert float(diag1['sigma1']) >= 1.-1e-12
        assert float(diag1['sigma2']) > .9999
        torch.testing.assert_close(model.from_average(1/(2*n-1)), out1, atol=1e-13, rtol=1e-13)
    assert float(model.q.weight.norm()) == pytest.approx(math.sqrt(2*math.log(n)))
    assert float(model.k.weight.norm()) == pytest.approx(math.sqrt(2*math.log(n)))


@pytest.mark.parametrize('candidates', [15, 63, 255])
def test_grover_formula_matches_dense_simulation_and_zero_mark_case(candidates):
    iterations, expected = grover_parameters(candidates)
    for marked in [0, candidates-1]:
        state = simulate_search_state(candidates, marked, iterations)
        assert np.square(state).sum() == pytest.approx(1.)
        assert state[marked]**2 == pytest.approx(expected, abs=1e-12)
    no_mark = simulate_search_state(candidates, None, iterations)
    np.testing.assert_allclose(no_mark, 1/math.sqrt(candidates), rtol=1e-12, atol=1e-12)
    assert expected > 2/3

