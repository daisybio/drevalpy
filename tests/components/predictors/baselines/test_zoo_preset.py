"""Tests for zoo-backed sklearn adapter scope."""

import pytest

from drevalpy.components.predictors.baselines.sklearn_base import SklearnModel
from drevalpy.components.predictors.baselines.zoo_preset import (
    MultiViewLightGBM,
    MultiViewRandomForest,
    MultiViewXGBoost,
    ZooPresetSklearnModel,
)
from drevalpy.models.single_drug import SingleDrugModelMixin


@pytest.mark.parametrize(
    "model_class",
    [ZooPresetSklearnModel, MultiViewRandomForest, MultiViewXGBoost, MultiViewLightGBM],
)
def test_zoo_preset_models_are_multi_drug(model_class: type) -> None:
    """Zoo-backed adapters use the default scope without the mixin."""
    assert issubclass(model_class, SklearnModel)
    assert not issubclass(model_class, SingleDrugModelMixin)
