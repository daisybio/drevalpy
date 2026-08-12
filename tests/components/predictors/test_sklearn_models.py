"""Tests for scikit-learn predictor component scope contracts."""

from __future__ import annotations

import pytest

from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.single_drug_sklearn import SingleDrugSklearnPredictor
from drevalpy.components.predictors.sklearn_models import (
    AdaBoostPredictor,
    ElasticNetPredictor,
    RandomForestPredictor,
    RidgePredictor,
    SingleDrugElasticNetPredictor,
    SingleDrugRandomForestPredictor,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.models.config import ModelConfig, from_spec
from drevalpy.registry._builtins import register_builtin_components
from drevalpy.types.enums.model_scope import ModelScope


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


@pytest.mark.parametrize("predictor_class", [ElasticNetPredictor, RandomForestPredictor])
def test_multi_drug_sklearn_predictors_are_multi_drug(predictor_class: type[Predictor]) -> None:
    assert predictor_class.scope is ModelScope.MULTI_DRUG


@pytest.mark.parametrize(
    ("predictor_class", "shared_predictor_class"),
    [
        (SingleDrugElasticNetPredictor, ElasticNetPredictor),
        (SingleDrugRandomForestPredictor, RandomForestPredictor),
    ],
)
def test_single_drug_sklearn_predictors_route_by_identity(
    predictor_class: type[Predictor],
    shared_predictor_class: type[Predictor],
) -> None:
    assert issubclass(predictor_class, shared_predictor_class)
    assert issubclass(predictor_class, SingleDrugSklearnPredictor)
    assert predictor_class.scope is ModelScope.SINGLE_DRUG


def test_ridge_zoo_preset_exists() -> None:
    config = from_spec("Ridge")
    assert isinstance(config, ModelConfig)
    assert config.predictor.name == "ridge"


def test_adaboost_default_depth_matches_space() -> None:
    predictor = AdaBoostPredictor()
    estimator = predictor._make_estimator()
    assert estimator.estimator.max_depth == 4


def test_sklearn_set_state_raises_when_estimator_missing() -> None:
    predictor = RidgePredictor()
    with pytest.raises(PredictorStateError):
        predictor.set_state({"hyperparameters": {"alpha": 1.0}, "mode": "regression"})
