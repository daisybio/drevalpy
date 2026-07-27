"""Single-drug scope contracts for zoo facades and ModelScope."""

from __future__ import annotations

import warnings

from drevalpy.models import construct_model
from drevalpy.models.config import ModelScope
from drevalpy.models.zoo import get_zoo_config


def _factory_tables():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from drevalpy.models import MODEL_FACTORY, SINGLE_DRUG_MODEL_FACTORY

        return MODEL_FACTORY, SINGLE_DRUG_MODEL_FACTORY


def test_single_drug_factory_membership() -> None:
    model_factory, single_drug_factory = _factory_tables()
    assert set(single_drug_factory) == {
        "SingleDrugElasticNet",
        "SingleDrugRandomForest",
        "MOLIR",
        "SuperFELTR",
    }
    for name, model_class in single_drug_factory.items():
        assert model_class.is_single_drug() is True
        assert model_factory[name] is model_class
        assert construct_model(name) is model_class
        assert get_zoo_config(name).scope == ModelScope.SINGLE_DRUG


def test_multi_drug_factory_excludes_single_drug_scope() -> None:
    model_factory, single_drug_factory = _factory_tables()
    for name in model_factory:
        if name in single_drug_factory:
            continue
        assert model_factory[name].is_single_drug() is False
        assert get_zoo_config(name).scope == ModelScope.MULTI_DRUG
