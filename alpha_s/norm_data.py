"""Validate the embedding norms stored by the measurement script."""

import numpy as np


def validate_embedding_norms(data, *, source="norm data", rtol=1e-5, check_spectral=True):
    """Reject impossible norms; passing this check does not prove sample alignment."""
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 4:
        raise ValueError(f"{source}: expected nonempty rows with at least four columns")

    # Validate the unplotted first-layer norm too; older random runs stored NaNs.
    values = data[:, :4]
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: token lengths and all norms must be finite")
    if np.any(data[:, 0] <= 0) or np.any(data[:, 1:4] < 0):
        raise ValueError(f"{source}: token lengths must be positive and norms nonnegative")
    if np.any(data[:, 0] != np.floor(data[:, 0])):
        raise ValueError(f"{source}: token lengths must be integers")

    invalid = np.flatnonzero(data[:, 3] > data[:, 1] * (1 + rtol))
    if check_spectral and invalid.size:
        rows = ", ".join(str(int(row) + 1) for row in invalid[:5])
        raise ValueError(
            f"{source}: spectral norm exceeds Frobenius norm in {invalid.size} "
            f"rows (first row numbers: {rows}). Regenerate the measurements "
            "with the corrected alpha_s.py before plotting; do not clip or drop rows."
        )
    if check_spectral and np.any(data[:, 3] * (1 + rtol) < data[:, 1] / np.sqrt(data[:, 0])):
        raise ValueError(f"{source}: spectral norm is below Frobenius norm / sqrt(N)")
    return data


def load_norm_data(path, *, check_spectral=True):
    data = np.load(path, allow_pickle=False)
    validate_embedding_norms(data, source=str(path), check_spectral=check_spectral)
    return data
