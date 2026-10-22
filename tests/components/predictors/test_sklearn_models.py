"""Tests for scikit-learn predictor component scope contracts."""

from __future__ import annotations

import pytest

from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.single_drug_sklearn import SingleDrugSklearnPredictor
from drevalpy.components.predictors.sklearn_models import (
    AdaBoostPredictor,
    ElasticNetPredictor,
    RandomForestPredictor,
    SingleDrugElasticNetPredictor,
    SingleDrugRandomForestPredictor,
)
from drevalpy.models import ModelSpec
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


def test_elasticnet_default_alpha_is_a_tenth_with_an_unchanged_search_range() -> None:
    space = ElasticNetPredictor.get_hyperparameter_space()

    assert ElasticNetPredictor.get_default_hyperparameters() == {"alpha": 0.1, "l1_ratio": 0.5}
    assert (space["alpha"]["low"], space["alpha"]["high"], space["alpha"]["log"]) == (1e-4, 10.0, True)
    assert ElasticNetPredictor()._make_estimator().alpha == 0.1


@pytest.mark.parametrize("zoo_name", ["ElasticNet", "SingleDrugElasticNet"])
def test_elasticnet_zoo_presets_do_not_override_alpha(zoo_name: str) -> None:
    resolved = ModelSpec.from_name(zoo_name).resolve()

    assert resolved.predictor_values()["alpha"] == 0.1


@pytest.mark.slow
def test_untuned_zoo_elasticnet_does_not_predict_a_constant_on_the_synthetic_dataset() -> None:
    import numpy as np

    import drevalpy
    from drevalpy.models import DRPModel

    dataset = drevalpy.synthetic()
    fold = drevalpy.data.split(dataset, "LPO", n_splits=2)[0]
    model = DRPModel(ModelSpec.from_name("ElasticNet"))

    model.train(dataset, fold.train)
    predictions = model.predict(dataset, fold.test)

    assert np.nanstd(predictions) > 0.05
