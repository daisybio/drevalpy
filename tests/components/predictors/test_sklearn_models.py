"""Tests for scikit-learn predictor component scope contracts."""

import pytest

from drevalpy.components.predictors.sklearn_models import (
    ElasticNetPredictor,
    RandomForestPredictor,
    SingleDrugElasticNetPredictor,
    SingleDrugRandomForestPredictor,
)


@pytest.mark.parametrize("predictor_class", [ElasticNetPredictor, RandomForestPredictor])
def test_multi_drug_sklearn_predictors_require_drug_features(predictor_class: type) -> None:
    assert predictor_class.requires_drug_featurizer is True


@pytest.mark.parametrize(
    ("predictor_class", "shared_predictor_class"),
    [
        (SingleDrugElasticNetPredictor, ElasticNetPredictor),
        (SingleDrugRandomForestPredictor, RandomForestPredictor),
    ],
)
def test_single_drug_sklearn_predictors_are_cell_line_only(
    predictor_class: type,
    shared_predictor_class: type,
) -> None:
    assert issubclass(predictor_class, shared_predictor_class)
    assert predictor_class.requires_drug_featurizer is False
