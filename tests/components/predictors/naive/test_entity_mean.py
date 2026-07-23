"""Tests for per-entity naive mean predictors."""

from __future__ import annotations

import numpy as np

from drevalpy.components.predictors.naive.entity_mean import NaiveDrugMeanPredictor
from tests.components.predictors.naive._helpers import naive_batch


def test_naive_drug_mean_predictor_uses_drug_fallback() -> None:
    predictor = NaiveDrugMeanPredictor()
    batch = naive_batch(
        cell_line_ids=np.array(["cl1", "cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1"]),
        response=np.array([1.0, 3.0, 5.0]),
    )
    predictor.fit(batch)
    preds = predictor.predict(
        naive_batch(
            cell_line_ids=np.array(["cl1", "cl2"]),
            drug_ids=np.array(["d1", "d2"]),
        )
    )
    assert preds.tolist() == [3.0, 3.0]
