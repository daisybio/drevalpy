"""Single-drug scope contracts for root facades and mixin marker."""

from __future__ import annotations

from drevalpy.models import (
    MODEL_FACTORY,
    SINGLE_DRUG_MODEL_FACTORY,
    SingleDrugModelMixin,
)


def test_single_drug_factory_membership() -> None:
    assert set(SINGLE_DRUG_MODEL_FACTORY) == {
        "SingleDrugElasticNet",
        "SingleDrugRandomForest",
        "MOLIR",
        "SuperFELTR",
    }
    for name, model_class in SINGLE_DRUG_MODEL_FACTORY.items():
        assert model_class.is_single_drug_model is True
        assert MODEL_FACTORY[name] is model_class


def test_single_drug_mixin_still_marks_scope() -> None:
    class EarlyStoppingSingleDrugModel(SingleDrugModelMixin):
        early_stopping = True

    assert EarlyStoppingSingleDrugModel.is_single_drug_model is True
    assert EarlyStoppingSingleDrugModel.early_stopping is True
