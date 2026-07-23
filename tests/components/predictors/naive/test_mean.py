"""Tests for global naive mean predictor."""

from __future__ import annotations

import numpy as np

from drevalpy.components.predictors.naive.mean import NaiveMeanPredictor
from tests.components.predictors.naive._helpers import naive_batch


def test_naive_mean_predictor_returns_dataset_mean() -> None:
    predictor = NaiveMeanPredictor()
    batch = naive_batch(
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
        response=np.array([2.0, 4.0]),
    )
    predictor.fit(batch)
    preds = predictor.predict(batch)
    assert preds.tolist() == [3.0, 3.0]
