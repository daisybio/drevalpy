"""Tests for built-in model zoo under drevalpy.models."""

from __future__ import annotations

import pytest

from drevalpy.models.factory import (
    naive_model_config,
    sklearn_model_config,
    sklearn_model_config_from_zoo,
)
from drevalpy.models.zoo import get_zoo_config, list_zoo_names, zoo_model_config


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
    assert zoo_config.cell_line_featurizer is not None
    assert factory_config.cell_line_featurizer is not None
    assert zoo_config.cell_line_featurizer.name == factory_config.cell_line_featurizer.name
    assert zoo_config.drug_featurizer is not None
    assert factory_config.drug_featurizer is not None
    assert zoo_config.drug_featurizer.name == factory_config.drug_featurizer.name
    assert zoo_config.predictor.name == factory_config.predictor.name


def test_zoo_naive_matches_factory() -> None:
    zoo_config = get_zoo_config("NaivePredictor")
    factory_config = naive_model_config("naiveMean")
    assert zoo_config.predictor.name == factory_config.predictor.name


def test_zoo_model_config_merges_hyperparameters() -> None:
    config = zoo_model_config("ElasticNet", {"alpha": 0.25})
    assert config.predictor.hyperparameters["alpha"] == 0.25


def test_sklearn_model_config_from_zoo_uses_zoo_entry() -> None:
    config = sklearn_model_config_from_zoo("ElasticNet", {"alpha": 0.5})
    assert config.predictor.name == "elasticNet"
    assert config.predictor.hyperparameters["alpha"] == 0.5


def test_single_drug_sklearn_zoo_entries_use_optional_drug_predictors() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    elastic_net = get_zoo_config("SingleDrugElasticNet")
    random_forest = get_zoo_config("SingleDrugRandomForest")

    assert elastic_net.predictor.name == "singleDrugElasticNet"
    assert random_forest.predictor.name == "singleDrugRandomForest"
    assert elastic_net.drug_featurizer is None
    assert random_forest.drug_featurizer is None
    elastic_net.validate()
    random_forest.validate()


def test_multi_drug_sklearn_predictor_without_drug_featurizer_fails() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = get_zoo_config("ElasticNet").model_copy(update={"drug_featurizer": None})

    with pytest.raises(ValueError, match="requires a drug_featurizer"):
        config.validate()
