"""Tests for internal model lookup helpers and modern construct_model forms."""

from __future__ import annotations

import warnings

import pytest

from drevalpy.models import construct_model
from drevalpy.models._model_lookup import (
    get_model_class,
    is_multi_drug_model_name,
    is_single_drug_model_name,
    known_model_names,
    multi_drug_model_names,
    single_drug_model_names,
)
from drevalpy.models.zoo import list_zoo_names
from drevalpy.types.enums.model_scope import ModelScope


def test_construct_model_one_arg_resolves_zoo_preset() -> None:
    model_cls = construct_model("ElasticNet")
    assert model_cls.get_model_name() == "ElasticNet"
    assert construct_model("ElasticNet") is model_cls


def test_construct_model_one_arg_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model spec"):
        construct_model("NotARealModel")


def test_list_zoo_names_scope_filter() -> None:
    single = list_zoo_names(include_external=False, scope=ModelScope.SINGLE_DRUG)
    multi = list_zoo_names(include_external=False, scope=ModelScope.MULTI_DRUG)
    assert "MOLIR" in single
    assert "SingleDrugElasticNet" in single
    assert "ElasticNet" in multi
    assert "ElasticNet" not in single
    assert set(single).isdisjoint(multi)
    assert set(single) | set(multi) == set(list_zoo_names(include_external=False))


def test_model_lookup_helpers_match_scope() -> None:
    assert is_single_drug_model_name("MOLIR")
    assert not is_single_drug_model_name("ElasticNet")
    assert is_multi_drug_model_name("ElasticNet")
    assert not is_multi_drug_model_name("NotARealModel")
    assert "MOLIR" in single_drug_model_names(include_external=False)
    assert "ElasticNet" in multi_drug_model_names(include_external=False)
    assert get_model_class("NaivePredictor").get_model_name() == "NaivePredictor"
    assert "ElasticNet" in known_model_names(include_external=False)


def test_factory_dict_import_emits_future_warning() -> None:
    import drevalpy.models as models

    # Ensure lazy __getattr__ runs again even if an earlier test already bound the dict.
    models.__dict__.pop("MODEL_FACTORY", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        factory = models.MODEL_FACTORY

    assert "ElasticNet" in factory
    assert any(issubclass(w.category, FutureWarning) and "MODEL_FACTORY" in str(w.message) for w in caught)


def test_construct_model_does_not_warn() -> None:
    import drevalpy.models as models

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = models.construct_model
        _ = construct_model("ElasticNet")

    assert not any(issubclass(w.category, FutureWarning) for w in caught)
