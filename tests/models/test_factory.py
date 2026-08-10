"""Tests for factory/zoo name resolution parity with configure flat HP."""

from __future__ import annotations

import pytest

from drevalpy.components.core.tuning.public_flat import config_from_public_hyperparameters
from drevalpy.models import construct_model
from drevalpy.models.config import ResolvedModelConfig
from drevalpy.models.factory import model_config_for_name


def test_model_config_for_name_matches_configure_path_for_predictor_hp() -> None:
    model_cls = construct_model("MultiViewRandomForest")
    flat = {"n_estimators": 8}
    via_factory = model_config_for_name("MultiViewRandomForest", flat)
    via_configure = config_from_public_hyperparameters(model_cls, flat)
    assert via_configure is not None
    assert isinstance(via_factory, ResolvedModelConfig)
    assert isinstance(via_configure, ResolvedModelConfig)
    assert via_factory.template.cell_line_featurizer is not None
    assert via_configure.template.cell_line_featurizer is not None
    assert via_factory.template.cell_line_featurizer.name == via_configure.template.cell_line_featurizer.name
    assert via_factory.predictor_values()["n_estimators"] == 8
    assert via_configure.predictor_values()["n_estimators"] == 8


def test_model_config_for_name_forwards_prediction_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from drevalpy.components.registry import get_predictor
    from drevalpy.types.enums.prediction_mode import PredictionMode

    monkeypatch.setattr(get_predictor("elasticNet"), "supported_modes", frozenset(PredictionMode))
    config = model_config_for_name("ElasticNet", prediction_mode=PredictionMode.CLASSIFICATION)
    assert not isinstance(config, ResolvedModelConfig)
    assert config.prediction_mode == PredictionMode.CLASSIFICATION
