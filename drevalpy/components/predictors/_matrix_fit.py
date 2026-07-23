"""Shared validation helpers for matrix predictors."""

from __future__ import annotations

import numpy as np


def validate_matrix_fit(x: np.ndarray, y: np.ndarray, *, n_pairs: int) -> None:
    """Reject empty or misaligned design matrices before fitting."""
    if n_pairs == 0:
        return
    if len(x) == 0:
        msg = "Matrix predictors cannot fit on an empty feature matrix for non-empty responses"
        raise ValueError(msg)
    if x.shape[0] != len(y):
        msg = f"Feature matrix rows ({x.shape[0]}) must match response length ({len(y)})"
        raise ValueError(msg)
    if x.shape[0] != n_pairs:
        msg = f"Feature matrix rows ({x.shape[0]}) must match batch pair count ({n_pairs})"
        raise ValueError(msg)
