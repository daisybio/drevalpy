"""Tests for reusable single-drug model scope."""

import pytest

from drevalpy.components.predictors.baselines.singledrug_baselines import (
    SingleDrugElasticNet,
    SingleDrugRandomForest,
)
from drevalpy.components.predictors.literature.impl.molir import molir
from drevalpy.components.predictors.literature.impl.superfeltr import superfeltr
from drevalpy.components.predictors.literature.public_models import MOLIR, SuperFELTR
from drevalpy.models import SingleDrugModelMixin


@pytest.mark.parametrize(
    "model_class",
    [
        SingleDrugElasticNet,
        SingleDrugRandomForest,
        MOLIR,
        SuperFELTR,
        molir.MOLIR,
        superfeltr.SuperFELTR,
    ],
)
def test_single_drug_models_share_scope_mixin(model_class: type) -> None:
    """Sklearn and literature single-drug models share one marker behavior.

    :param model_class: concrete single-drug model or engine under test
    """
    assert issubclass(model_class, SingleDrugModelMixin)
    assert model_class.is_single_drug_model is True
    assert model_class.requires_drug_featurizer is False
    assert model_class.drug_views == []


def test_single_drug_mixin_does_not_prescribe_early_stopping() -> None:
    """Algorithms retain control of early-stopping behavior."""

    class EarlyStoppingSingleDrugModel(SingleDrugModelMixin):
        early_stopping = True

    assert EarlyStoppingSingleDrugModel.early_stopping is True
