"""Tests for tissue-aware naive predictors."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.naive.tissue import NaiveTissueMeanPredictor
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, TISSUE_IDENTIFIER
from tests.components.predictors.naive._helpers import naive_batch


def test_naive_tissue_mean_requires_tissue_features() -> None:
    predictor = NaiveTissueMeanPredictor()
    batch = naive_batch(
        cell_line_ids=np.array(["cl1"]),
        drug_ids=np.array(["d1"]),
        response=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="requires tissue"):
        predictor.fit(batch)


def test_naive_tissue_mean_predicts_per_tissue() -> None:
    cell_line_input = FeatureDataset(
        features={
            "cl1": {CELL_LINE_IDENTIFIER: np.array(["cl1"]), TISSUE_IDENTIFIER: np.array(["lung"])},
            "cl2": {CELL_LINE_IDENTIFIER: np.array(["cl2"]), TISSUE_IDENTIFIER: np.array(["blood"])},
        }
    )
    predictor = NaiveTissueMeanPredictor()
    batch = naive_batch(
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d1"]),
        response=np.array([2.0, 4.0]),
        cell_line_input=cell_line_input,
    )
    predictor.fit(batch)
    preds = predictor.predict(batch)
    assert preds.tolist() == [2.0, 4.0]
