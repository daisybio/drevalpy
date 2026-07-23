"""Tests for shared drevalpy type definitions."""

from __future__ import annotations

import drevalpy.models.config as model_config
import drevalpy.types.prediction_mode as prediction_mode_module
from drevalpy.types.prediction_mode import PredictionMode


def test_prediction_mode_values() -> None:
    assert PredictionMode.REGRESSION == "regression"
    assert PredictionMode.CLASSIFICATION == "classification"


def test_prediction_mode_reexported_from_models_config() -> None:
    assert model_config.PredictionMode is PredictionMode


def test_predictors_base_imports_prediction_mode_from_types() -> None:
    import drevalpy.components.predictors.base as base_module

    source_path = base_module.__file__
    assert source_path is not None
    text = open(source_path, encoding="utf-8").read()
    assert "drevalpy.types.prediction_mode" in text
    assert "drevalpy.models.config" not in text


def test_prediction_mode_module_has_no_models_dependency() -> None:
    source_path = prediction_mode_module.__file__
    assert source_path is not None
    text = open(source_path, encoding="utf-8").read()
    assert "drevalpy.models" not in text
