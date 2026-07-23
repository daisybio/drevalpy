"""Tests for naive mean-effects predictor."""

from __future__ import annotations

import numpy as np

from drevalpy.components.predictors.naive.effects import NaiveMeanEffectsPredictor
from tests.components.predictors.naive._helpers import naive_batch


def test_naive_mean_effects_without_tissue_decomposes_cell_and_drug() -> None:
    predictor = NaiveMeanEffectsPredictor()
    batch = naive_batch(
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
        response=np.array([1.0, 2.0, 5.0, 8.0]),
    )
    predictor.fit(batch)
    preds = predictor.predict(
        naive_batch(
            cell_line_ids=np.array(["cl1", "cl2"]),
            drug_ids=np.array(["d1", "d2"]),
        )
    )
    assert preds.shape == (2,)
    assert np.isfinite(preds).all()
