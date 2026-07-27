"""Tests for per-entity naive mean predictors."""

from __future__ import annotations

import numpy as np

from drevalpy.components.predictors.naive.entity_mean import (
    NaiveCellLineMeanPredictor,
    NaiveDrugMeanPredictor,
)
from tests.components.predictors.naive._helpers import naive_batch, one_hot


def test_naive_drug_mean_follows_matrix_not_ids() -> None:
    categories = ["d1", "d2"]
    # Entity matrix rows are [d1, d2]; pair index maps pairs onto those rows.
    drug_features = one_hot(categories, categories)
    predictor = NaiveDrugMeanPredictor()
    predictor.fit(
        naive_batch(
            response=np.array([1.0, 3.0, 5.0]),
            drug_features=drug_features,
            drug_pair_idx=np.array([0, 1, 0], dtype=np.int64),
            cell_line_ids=np.array(["noise1", "noise2", "noise3"]),
            drug_ids=np.array(["not_d1", "not_d2", "still_not_d1"]),
        )
    )
    # Predictions must follow matrix columns, not the decoy drug_ids.
    preds = predictor.predict(
        naive_batch(
            drug_features=drug_features,
            drug_pair_idx=np.array([0, 1], dtype=np.int64),
            drug_ids=np.array(["decoy_a", "decoy_b"]),
        )
    )
    assert preds.tolist() == [3.0, 3.0]


def test_naive_cell_line_mean_unseen_zero_row_falls_back() -> None:
    categories = ["cl1", "cl2"]
    cell_features = one_hot(categories, categories)
    predictor = NaiveCellLineMeanPredictor()
    predictor.fit(
        naive_batch(
            response=np.array([2.0, 6.0]),
            cell_line_features=cell_features,
            cell_line_pair_idx=np.array([0, 1], dtype=np.int64),
        )
    )
    unseen = np.zeros((1, 2), dtype=np.float64)
    preds = predictor.predict(
        naive_batch(
            cell_line_features=np.vstack([cell_features, unseen]),
            cell_line_pair_idx=np.array([0, 1, 2], dtype=np.int64),
        )
    )
    assert preds.tolist() == [2.0, 6.0, 4.0]


def test_naive_drug_mean_state_roundtrip() -> None:
    categories = ["d1", "d2"]
    drug_features = one_hot(categories, categories)
    predictor = NaiveDrugMeanPredictor()
    fit_batch = naive_batch(
        response=np.array([1.0, 5.0]),
        drug_features=drug_features,
        drug_pair_idx=np.array([0, 1], dtype=np.int64),
    )
    predictor.fit(fit_batch)
    restored = NaiveDrugMeanPredictor()
    restored.set_state(predictor.get_state())
    preds = restored.predict(fit_batch)
    assert preds.tolist() == [1.0, 5.0]
