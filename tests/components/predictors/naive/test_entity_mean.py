"""Tests for per-entity naive mean predictors (lifecycle lives in ``test_single_entity``)."""

from __future__ import annotations

import numpy as np

from drevalpy.components.predictors.naive.entity_mean import NaiveDrugMeanPredictor, NaiveSampleMeanPredictor
from tests.components.predictors.naive._helpers import naive_batch, one_hot

CATEGORIES = ["e1", "e2"]


def test_drug_mean_follows_the_matrix_not_decoy_ids_and_round_trips() -> None:
    features = one_hot(CATEGORIES, CATEGORIES)
    predictor = NaiveDrugMeanPredictor()
    predictor.fit(
        naive_batch(
            response=np.array([1.0, 3.0, 5.0]),
            drug_features=features,
            drug_pair_idx=np.array([0, 1, 0], dtype=np.int64),
            drug_ids=np.array(["not_d1", "not_d2", "still_not_d1"]),
        )
    )
    batch = naive_batch(
        drug_features=features,
        drug_pair_idx=np.array([0, 1], dtype=np.int64),
        drug_ids=np.array(["decoy_a", "decoy_b"]),
    )
    restored = NaiveDrugMeanPredictor()
    restored.set_state(predictor.get_state())

    assert predictor.predict(batch).tolist() == [3.0, 3.0]
    assert restored.predict(batch).tolist() == [3.0, 3.0]


def test_sample_mean_falls_back_for_an_unseen_zero_row() -> None:
    features = one_hot(CATEGORIES, CATEGORIES)
    predictor = NaiveSampleMeanPredictor()
    predictor.fit(
        naive_batch(
            response=np.array([2.0, 6.0]),
            sample_features=features,
            sample_pair_idx=np.array([0, 1], dtype=np.int64),
        )
    )
    preds = predictor.predict(
        naive_batch(
            sample_features=np.vstack([features, np.zeros((1, 2))]),
            sample_pair_idx=np.array([0, 1, 2], dtype=np.int64),
        )
    )

    assert preds.tolist() == [2.0, 6.0, 4.0]
