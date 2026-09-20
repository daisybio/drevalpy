"""Tests for zoo name to ModelSpec configuration resolution."""

from __future__ import annotations

import pytest

from drevalpy.models import ModelSpec
from drevalpy.models.config import ResolvedModelConfig
from drevalpy.models.factory import model_config_for_name
from drevalpy.types.enums.prediction_mode import PredictionMode


def test_model_config_for_name_matches_spec_resolution() -> None:
    flat = {"n_estimators": 8}
    via_factory = model_config_for_name("MultiViewRandomForest", flat)
    via_spec = ModelSpec.from_name("MultiViewRandomForest").resolve(flat)

    assert isinstance(via_factory, ResolvedModelConfig)
    assert via_factory.predictor_values()["n_estimators"] == via_spec.predictor_values()["n_estimators"] == 8


def test_model_config_for_name_forwards_prediction_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from drevalpy.registry.predictor import get as get_predictor

    monkeypatch.setattr(get_predictor("elasticNet"), "supported_modes", frozenset(PredictionMode))
    config = model_config_for_name("ElasticNet", prediction_mode=PredictionMode.CLASSIFICATION)
    assert not isinstance(config, ResolvedModelConfig)
    assert config.prediction_mode == PredictionMode.CLASSIFICATION
