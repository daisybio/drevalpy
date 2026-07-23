"""Tests for scikit-learn predictor component scope contracts."""

import pytest

from drevalpy.components.predictors.sklearn_models import (
    ElasticNetPredictor,
    RandomForestPredictor,
    SingleDrugElasticNetPredictor,
    SingleDrugRandomForestPredictor,
)
from drevalpy.models.single_drug import SingleDrugModelMixin


@pytest.mark.parametrize("predictor_class", [ElasticNetPredictor, RandomForestPredictor])
def test_multi_drug_sklearn_predictors_require_drug_features(predictor_class: type) -> None:
    """Multi-drug predictor components retain the default drug-feature contract.

    :param predictor_class: concrete multi-drug predictor under test
    """
    assert predictor_class.requires_drug_featurizer is True
    assert not issubclass(predictor_class, SingleDrugModelMixin)


@pytest.mark.parametrize(
    ("predictor_class", "shared_predictor_class"),
    [
        (SingleDrugElasticNetPredictor, ElasticNetPredictor),
        (SingleDrugRandomForestPredictor, RandomForestPredictor),
    ],
)
def test_single_drug_sklearn_predictors_use_scope_mixin(
    predictor_class: type,
    shared_predictor_class: type,
) -> None:
    """Single-drug variants reuse estimators while making drug features optional.

    :param predictor_class: concrete single-drug predictor under test
    :param shared_predictor_class: multi-drug class providing its estimator behavior
    """
    assert issubclass(predictor_class, shared_predictor_class)
    assert issubclass(predictor_class, SingleDrugModelMixin)
    assert predictor_class.requires_drug_featurizer is False
