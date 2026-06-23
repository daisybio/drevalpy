"""Tests for built-in zoo entries."""

from __future__ import annotations

from drevalpy.components.factory import (
    naive_model_config,
    sklearn_model_config,
    sklearn_model_config_from_zoo,
)
from drevalpy.components.zoo import get_zoo_config, list_zoo_names, zoo_model_config


def test_builtin_zoo_lists_passing_models() -> None:
    names = list_zoo_names(include_external=False)
    assert "ElasticNet" in names
    assert "NaivePredictor" in names
    assert "DIPK" in names
    assert "PharmaFormer" in names
    assert "SingleDrugElasticNet" in names


def test_zoo_elastic_net_matches_factory_defaults() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    zoo_config = get_zoo_config("ElasticNet")
    factory_config = sklearn_model_config("elasticNet", {})
    assert zoo_config.cell_line_featurizer.type == factory_config.cell_line_featurizer.type
    assert zoo_config.drug_featurizer.type == factory_config.drug_featurizer.type
    assert zoo_config.predictor.type == factory_config.predictor.type


def test_zoo_naive_matches_factory() -> None:
    zoo_config = get_zoo_config("NaivePredictor")
    factory_config = naive_model_config("naiveMean")
    assert zoo_config.predictor.type == factory_config.predictor.type


def test_zoo_model_config_merges_hyperparameters() -> None:
    config = zoo_model_config("ElasticNet", {"alpha": 0.25})
    assert config.predictor.hyperparameters["alpha"] == 0.25


def test_sklearn_model_config_from_zoo_uses_zoo_entry() -> None:
    config = sklearn_model_config_from_zoo("ElasticNet", {"alpha": 0.5})
    assert config.predictor.type == "elasticNet"
    assert config.predictor.hyperparameters["alpha"] == 0.5
