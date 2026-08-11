"""Tests for tissue-aware naive predictors."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.naive.tissue import (
    NaiveTissueDrugMeanPredictor,
    NaiveTissueMeanPredictor,
)
from tests.components.predictors.naive._helpers import naive_batch, one_hot


def test_naive_tissue_mean_requires_tissue_features() -> None:
    predictor = NaiveTissueMeanPredictor()
    batch = naive_batch(
        response=np.array([1.0]),
        cell_line_features=np.empty((1, 0), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="requires tissue"):
        predictor.fit(batch)


def test_naive_tissue_mean_predicts_from_matrix_not_ids() -> None:
    tissues = ["lung", "blood"]
    tissue_features = one_hot(tissues, tissues)
    predictor = NaiveTissueMeanPredictor()
    predictor.fit(
        naive_batch(
            response=np.array([2.0, 4.0]),
            cell_line_features=tissue_features,
            cell_line_pair_idx=np.array([0, 1], dtype=np.int64),
            cell_line_ids=np.array(["ignore_a", "ignore_b"]),
        )
    )
    preds = predictor.predict(
        naive_batch(
            cell_line_features=tissue_features,
            cell_line_pair_idx=np.array([0, 1], dtype=np.int64),
            cell_line_ids=np.array(["other_a", "other_b"]),
        )
    )
    assert preds.tolist() == [2.0, 4.0]


def test_naive_tissue_drug_mean_interaction_table() -> None:
    tissue_cats = ["lung", "blood"]
    drug_cats = ["d1", "d2"]
    # Pairs: lung-d1=1, lung-d2=3, blood-d1=5  (blood-d2 unseen)
    tissue_entity = one_hot(tissue_cats, tissue_cats)
    drug_entity = one_hot(drug_cats, drug_cats)
    predictor = NaiveTissueDrugMeanPredictor()
    predictor.fit(
        naive_batch(
            response=np.array([1.0, 3.0, 5.0]),
            cell_line_features=tissue_entity,
            drug_features=drug_entity,
            cell_line_pair_idx=np.array([0, 0, 1], dtype=np.int64),
            drug_pair_idx=np.array([0, 1, 0], dtype=np.int64),
        )
    )
    dataset_mean = 3.0
    preds = predictor.predict(
        naive_batch(
            cell_line_features=tissue_entity,
            drug_features=drug_entity,
            cell_line_pair_idx=np.array([0, 0, 1, 1], dtype=np.int64),
            drug_pair_idx=np.array([0, 1, 0, 1], dtype=np.int64),
        )
    )
    # Unseen blood-d2 falls back to dataset mean.
    assert preds.tolist() == [1.0, 3.0, 5.0, dataset_mean]


def test_naive_tissue_mean_state_roundtrip() -> None:
    tissues = ["lung", "blood"]
    tissue_features = one_hot(tissues, tissues)
    predictor = NaiveTissueMeanPredictor()
    batch = naive_batch(
        response=np.array([2.0, 8.0]),
        cell_line_features=tissue_features,
        cell_line_pair_idx=np.array([0, 1], dtype=np.int64),
    )
    predictor.fit(batch)
    restored = NaiveTissueMeanPredictor()
    restored.set_state(predictor.get_state())
    np.testing.assert_allclose(restored.predict(batch), [2.0, 8.0])
