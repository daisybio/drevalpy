"""Tests for MOLIR utility helpers."""

from __future__ import annotations

import numpy as np

from drevalpy.components.predictors.literature.molir.utils import _realign_omic_matrix


def test_realign_omic_matrix_realigns_columns_and_fills_missing() -> None:
    model_features = ["g1", "g2"]
    incoming_features = ["g0", "g1", "g2"]
    values = np.array([[1.0, 2.0, 3.0]])

    result = _realign_omic_matrix(values, model_features, incoming_features)

    assert result.shape == (1, 2)
    np.testing.assert_allclose(result, [[2.0, 3.0]])


def test_realign_omic_matrix_reorders_columns() -> None:
    model_features = ["c2", "c1"]
    incoming_features = ["c0", "c1", "c2"]
    values = np.array([[6.0, 7.0, 8.0]])

    result = _realign_omic_matrix(values, model_features, incoming_features)

    assert result.shape == (1, 2)
    np.testing.assert_allclose(result, [[8.0, 7.0]])


def test_realign_omic_matrix_fills_missing_with_zeros() -> None:
    model_features = ["a", "b", "c"]
    incoming_features = ["b", "d"]
    values = np.array([[10.0, 20.0]])

    result = _realign_omic_matrix(values, model_features, incoming_features)

    assert result.shape == (1, 3)
    np.testing.assert_allclose(result, [[0.0, 10.0, 0.0]])
