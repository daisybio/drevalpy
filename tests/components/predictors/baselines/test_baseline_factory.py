"""Tests for baseline refactor: public factory and compatibility imports."""

from __future__ import annotations

import importlib

import pytest

from drevalpy.components.config import ModelConfig
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.models import MODEL_FACTORY

BASELINE_FACTORY_NAMES = [
    "NaivePredictor",
    "NaiveDrugMeanPredictor",
    "NaiveCellLineMeanPredictor",
    "NaiveMeanEffectsPredictor",
    "NaiveTissueMeanPredictor",
    "NaiveTissueDrugMeanPredictor",
    "ElasticNet",
    "RandomForest",
    "SVR",
    "GradientBoosting",
    "AdaBoostDecisionTree",
    "Lasso",
    "KNNRegressor",
    "MultiViewRandomForest",
    "MultiViewXGBoost",
    "SingleDrugElasticNet",
    "SingleDrugRandomForest",
]


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


@pytest.mark.parametrize("name", BASELINE_FACTORY_NAMES)
def test_model_factory_imports_component_baselines(name: str) -> None:
    cls = MODEL_FACTORY[name]
    assert cls.__module__.startswith("drevalpy.components.predictors.baselines")


@pytest.mark.parametrize("name", ["ElasticNet", "RandomForest", "NaivePredictor", "MultiViewRandomForest"])
def test_model_config_and_factory_share_name(name: str) -> None:
    config = ModelConfig.from_spec(name)
    model_cls = MODEL_FACTORY[name]
    config.validate()
    assert model_cls.get_model_name() == name


def test_legacy_baseline_import_paths_still_resolve() -> None:
    sklearn = importlib.import_module("drevalpy.models.baselines.sklearn_models")
    assert sklearn.ElasticNetModel.__module__.startswith("drevalpy.components.predictors.baselines")
    naive = importlib.import_module("drevalpy.models.baselines.naive_pred")
    assert naive.NaivePredictor.__module__.startswith("drevalpy.components.predictors.baselines")
