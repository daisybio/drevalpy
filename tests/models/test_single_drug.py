"""Single-drug scope contracts for zoo facades and ModelScope."""

from __future__ import annotations

from drevalpy.models import construct_model
from drevalpy.models.config import ModelScope
from drevalpy.models.zoo import get_zoo_config, list_zoo_names


def test_single_drug_zoo_membership() -> None:
    single_names = list_zoo_names(include_external=False, scope=ModelScope.SINGLE_DRUG)
    assert set(single_names) == {
        "SingleDrugElasticNet",
        "SingleDrugRandomForest",
        "MOLIR",
        "SuperFELTR",
    }
    for name in single_names:
        model_class = construct_model(name)
        assert model_class.is_single_drug() is True
        assert get_zoo_config(name).scope == ModelScope.SINGLE_DRUG


def test_multi_drug_zoo_excludes_single_drug_scope() -> None:
    multi_names = list_zoo_names(include_external=False, scope=ModelScope.MULTI_DRUG)
    for name in multi_names:
        model_class = construct_model(name)
        assert model_class.is_single_drug() is False
        assert get_zoo_config(name).scope == ModelScope.MULTI_DRUG
