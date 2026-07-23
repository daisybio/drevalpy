"""Tests for sklearn adapter scope bases."""

import pytest

from drevalpy.components.predictors.baselines.sklearn_base import (
    SingleDrugSklearnModel,
    SklearnModel,
)
from drevalpy.models.single_drug import SingleDrugModelMixin


def test_scope_bases_share_sklearn_adapter() -> None:
    """Single-drug sklearn behavior composes a reusable scope mixin."""
    assert issubclass(SingleDrugSklearnModel, SklearnModel)
    assert issubclass(SingleDrugSklearnModel, SingleDrugModelMixin)


def test_multi_drug_scope_requires_a_drug_view() -> None:
    """Multi-drug adapters reject recipes without drug features."""

    class ExampleMultiDrugModel(SklearnModel):
        @classmethod
        def get_model_name(cls) -> str:
            return "ElasticNet"

    model = ExampleMultiDrugModel()
    with pytest.raises(ValueError, match="require at least one drug view"):
        model.build_model({"drug_views": []})
