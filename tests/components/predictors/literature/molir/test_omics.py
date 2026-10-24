"""Tests for the lightning-free MOLIR omics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.literature.molir._omics import _realign_omic_matrix


@pytest.mark.parametrize(
    ("values", "model_features", "incoming_features", "expected"),
    [
        ([1.0, 2.0, 3.0], ["g1", "g2"], ["g0", "g1", "g2"], [2.0, 3.0]),
        ([6.0, 7.0, 8.0], ["c2", "c1"], ["c0", "c1", "c2"], [8.0, 7.0]),
        ([10.0, 20.0], ["a", "b", "c"], ["b", "d"], [0.0, 10.0, 0.0]),
        ([10.0, 20.0, 30.0], ["a", "b", "c"], ["c", "a", "b"], [20.0, 30.0, 10.0]),
        ([10.0, 20.0, 30.0], ["a", "b", "missing"], ["c", "a", "b"], [20.0, 30.0, 0.0]),
    ],
    ids=["narrow", "reorder", "zero-fill", "equal-width-permutation", "equal-width-missing"],
)
def test_realign_omic_matrix(values, model_features, incoming_features, expected) -> None:
    result = _realign_omic_matrix(np.array([values]), model_features, incoming_features)

    np.testing.assert_allclose(result, [expected])


def test_realign_omic_matrix_passes_matching_features_through_unchanged() -> None:
    values = np.array([[1.0, 2.0]])

    assert _realign_omic_matrix(values, ["a", "b"], ["a", "b"]) is values
