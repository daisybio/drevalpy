"""Tests for global naive mean predictor."""

from __future__ import annotations

import numpy as np

from drevalpy.components.predictors.naive.mean import NaiveMeanPredictor
from tests.components.predictors.naive._helpers import naive_batch


def test_naive_mean_predictor_returns_dataset_mean() -> None:
    predictor = NaiveMeanPredictor()
    batch = naive_batch(response=np.array([2.0, 4.0]))
    predictor.fit(batch)
    preds = predictor.predict(batch)
    assert preds.tolist() == [3.0, 3.0]


def test_naive_mean_predictor_ignores_ids_and_matrices() -> None:
    predictor = NaiveMeanPredictor()
    fit_batch = naive_batch(
        response=np.array([1.0, 5.0]),
        cell_line_ids=np.array(["wrong_a", "wrong_b"]),
        drug_ids=np.array(["wrong_x", "wrong_y"]),
    )
    predictor.fit(fit_batch)
    preds = predictor.predict(
        naive_batch(
            n_pairs=3,
            cell_line_ids=np.array(["a", "b", "c"]),
            drug_ids=np.array(["x", "y", "z"]),
        )
    )
    assert preds.tolist() == [3.0, 3.0, 3.0]


def test_naive_mean_state_roundtrip() -> None:
    predictor = NaiveMeanPredictor()
    predictor.fit(naive_batch(response=np.array([2.0, 6.0])))
    restored = NaiveMeanPredictor()
    restored.set_state(predictor.get_state())
    preds = restored.predict(naive_batch(n_pairs=2))
    assert preds.tolist() == [4.0, 4.0]
