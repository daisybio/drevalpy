"""Tests for FeatureFreePredictor."""

from __future__ import annotations

from drevalpy.components.predictors.base import Predictor
from drevalpy.components.predictors.feature_free import FeatureFreePredictor


def test_feature_free_predictor_defaults() -> None:
    assert issubclass(FeatureFreePredictor, Predictor)
    assert FeatureFreePredictor.requires_drug_featurizer is False
    assert FeatureFreePredictor.input_interface == "feature_free"
