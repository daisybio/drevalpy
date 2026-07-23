"""Single-drug scope contracts for zoo facades and ModelScope."""

from __future__ import annotations

from drevalpy.models import MODEL_FACTORY, SINGLE_DRUG_MODEL_FACTORY
from drevalpy.models.config import ModelScope
from drevalpy.models.zoo import get_zoo_config


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
        assert get_zoo_config(name).scope == ModelScope.SINGLE_DRUG


def test_multi_drug_factory_excludes_single_drug_scope() -> None:
    for name in MODEL_FACTORY:
        if name in SINGLE_DRUG_MODEL_FACTORY:
            continue
        assert MODEL_FACTORY[name].is_single_drug_model is False
        assert get_zoo_config(name).scope == ModelScope.MULTI_DRUG
