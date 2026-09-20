"""Tests for model-name lookup and public specification resolution."""

from __future__ import annotations

from drevalpy.models import ModelSpec
from drevalpy.models._model_lookup import known_model_names, single_drug_model_names
from drevalpy.models.zoo import list_zoo_names
from drevalpy.types.enums.model_scope import ModelScope


def test_model_spec_resolves_zoo_preset() -> None:
    spec = ModelSpec.from_name("ElasticNet")
    assert spec.name == "ElasticNet"
    assert spec.config.predictor.name == "elasticNet"


def test_list_zoo_names_scope_filter() -> None:
    single = list_zoo_names(include_external=False, scope=ModelScope.SINGLE_DRUG)
    multi = list_zoo_names(include_external=False, scope=ModelScope.MULTI_DRUG)
    assert "MOLIR" in single
    assert "SingleDrugElasticNet" in single
    assert "ElasticNet" in multi
    assert "ElasticNet" not in single
    assert set(single).isdisjoint(multi)
    assert set(single) | set(multi) == set(list_zoo_names(include_external=False))


def test_model_lookup_helpers() -> None:
    assert "MOLIR" in single_drug_model_names(include_external=False)
    assert "ElasticNet" in known_model_names(include_external=False)


def test_factory_names_include_builtin_presets() -> None:
    names = set(known_model_names(include_external=False))
    for name in ("NaivePredictor", "NaiveDrugMeanPredictor", "ElasticNet", "RandomForest", "SVR", "MOLIR"):
        assert name in names
