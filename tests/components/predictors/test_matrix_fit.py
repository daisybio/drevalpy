"""Tests for matrix fit validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors._matrix_fit import validate_matrix_fit


def test_validate_matrix_fit_rejects_row_mismatch() -> None:
    x = np.ones((3, 2), dtype=np.float32)
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Feature matrix rows"):
        validate_matrix_fit(x, y, n_pairs=3)


def test_validate_matrix_fit_rejects_empty_features_for_non_empty_batch() -> None:
    x = np.empty((0, 2), dtype=np.float32)
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="empty feature matrix"):
        validate_matrix_fit(x, y, n_pairs=2)
