"""Tests for the global naive mean predictor."""

from __future__ import annotations

import numpy as np

from drevalpy.components.predictors.naive.mean import NaiveMeanPredictor
from tests.components.predictors.naive._helpers import naive_batch


def test_predicts_the_dataset_mean_for_any_batch_and_round_trips_state() -> None:
    predictor = NaiveMeanPredictor()
    predictor.fit(naive_batch(response=np.array([1.0, 5.0]), sample_ids=np.array(["x", "y"])))
    restored = NaiveMeanPredictor()
    restored.set_state(predictor.get_state())

    assert predictor.predict(naive_batch(n_pairs=3, sample_ids=np.array(["a", "b", "c"]))).tolist() == [3.0] * 3
    assert restored.predict(naive_batch(n_pairs=2)).tolist() == [3.0, 3.0]
