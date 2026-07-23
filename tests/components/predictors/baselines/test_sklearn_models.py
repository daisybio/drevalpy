"""Tests for concrete multi-drug sklearn adapters."""

import pytest

from drevalpy.components.predictors.baselines.sklearn_base import SklearnModel
from drevalpy.components.predictors.baselines.sklearn_models import (
    AdaBoostDecisionTree,
    ElasticNetModel,
    GradientBoosting,
    KNNRegressor,
    LassoModel,
    RandomForest,
    SVMRegressor,
)
from drevalpy.models.single_drug import SingleDrugModelMixin


@pytest.mark.parametrize(
    "model_class",
    [
        AdaBoostDecisionTree,
        ElasticNetModel,
        GradientBoosting,
        KNNRegressor,
        LassoModel,
        RandomForest,
        SVMRegressor,
    ],
)
def test_concrete_sklearn_models_use_default_multi_drug_scope(model_class: type) -> None:
    """Standard sklearn adapters inherit the default scope without the mixin."""
    assert issubclass(model_class, SklearnModel)
    assert not issubclass(model_class, SingleDrugModelMixin)
