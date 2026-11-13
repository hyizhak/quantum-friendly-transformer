"""Reject impossible archived measurements before producing spectral-norm plots."""

import numpy as np
import pytest

from alpha_s.norm_data import load_norm_data, validate_embedding_norms


def test_valid_norms_round_trip(tmp_path):
    data = np.array([[2, 5, 10, 4], [1, 3, 6, 3], [4, 0, 0, 0]], dtype=float)
    path = tmp_path / "norms.npy"
    np.save(path, data)
    np.testing.assert_array_equal(load_norm_data(path), data)


def test_mismatched_spectral_norm_is_rejected_without_rewriting_data(tmp_path):
    data = np.array([[18, 4.726562, 7.736837, 4.692175],
                     [32, 0.756293, 6.195638, 4.692175]])
    path = tmp_path / "mistral7b_data.npy"
    np.save(path, data)
    original = path.read_bytes()
    with pytest.raises(ValueError, match=r"mistral7b_data.npy: spectral norm.*row numbers: 2"):
        load_norm_data(path)
    assert path.read_bytes() == original
    # Frobenius-only plots do not consume the corrupted spectral column.
    np.testing.assert_array_equal(load_norm_data(path, check_spectral=False), data)


@pytest.mark.parametrize("data, message", [
    ([], "four columns"),
    ([[1, 2, 3]], "four columns"),
    ([[1, np.nan, 3, 1]], "finite"),
    ([[1, 2, 3, np.inf]], "finite"),
    ([[1, 2, np.nan, 1]], "finite"),
    ([[1.5, 2, 3, 1]], "integers"),
    ([[0, 2, 3, 1]], "positive"),
    ([[1, 2, 3, -1]], "nonnegative"),
    ([[4, 4, 3, .1]], "below Frobenius"),
])
def test_invalid_embedding_norms_are_rejected(data, message):
    with pytest.raises(ValueError, match=message):
        validate_embedding_norms(data)


def test_float_roundoff_at_equal_norms_is_allowed():
    validate_embedding_norms([[1, 1.0, 2.0, 1.0 + 1e-7]])
