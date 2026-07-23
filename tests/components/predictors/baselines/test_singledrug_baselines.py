"""Tests for concrete single-drug sklearn adapters."""

import pytest

from drevalpy.components.predictors.baselines.singledrug_baselines import (
    SingleDrugElasticNet,
    SingleDrugRandomForest,
)
from drevalpy.components.predictors.baselines.sklearn_base import SingleDrugSklearnModel
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.models.single_drug import SingleDrugModelMixin


@pytest.mark.parametrize("model_class", [SingleDrugElasticNet, SingleDrugRandomForest])
def test_single_drug_models_share_scope_behavior(model_class: type[SingleDrugSklearnModel]) -> None:
    """Single-drug adapters always omit drug features."""
    register_builtin_components()
    assert issubclass(model_class, SingleDrugSklearnModel)
    assert issubclass(model_class, SingleDrugModelMixin)

    model = model_class()
    model.build_model(
        {
            "cell_line_views": ["gene_expression"],
            "drug_views": ["fingerprints"],
        }
    )

    assert model.is_single_drug_model
    assert model.drug_views == []
    assert model.load_drug_features("/unused", "unused") is None
