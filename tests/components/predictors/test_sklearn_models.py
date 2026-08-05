"""Tests for scikit-learn predictor component scope contracts."""

from __future__ import annotations

import pytest

from drevalpy.components.predictors.base import Predictor
from drevalpy.components.predictors.single_drug_sklearn import SingleDrugSklearnPredictor
from drevalpy.components.predictors.sklearn_models import (
    ElasticNetPredictor,
    RandomForestPredictor,
    SingleDrugElasticNetPredictor,
    SingleDrugRandomForestPredictor,
)


@pytest.mark.parametrize("predictor_class", [ElasticNetPredictor, RandomForestPredictor])
def test_multi_drug_sklearn_predictors_require_drug_features(predictor_class: type[Predictor]) -> None:
    assert predictor_class.requires_drug_featurizer is True


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
    assert predictor_class.requires_drug_featurizer is True
    assert predictor_class.routing_drug_featurizer == "identity"
