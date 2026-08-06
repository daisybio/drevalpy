"""Tests for immutable ModelConfig and featurizer/predictor templates."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.models.config import (
    CellLineFeaturizerConfig,
    DrugFeaturizerConfig,
    FeaturizerConfig,
    ModelConfig,
    PredictorConfig,
)
from drevalpy.models.zoo import get_zoo_config


def test_model_config_rejects_field_assignment() -> None:
    register_builtin_components()
    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="scaledGeneExpression"),
        drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
        predictor=PredictorConfig(name="elasticNet"),
    )
    with pytest.raises(ValidationError, match="frozen"):
        config.scope = "single_drug"  # type: ignore[misc]


def test_featurizer_config_rejects_deep_options_mutation() -> None:
    register_builtin_components()
    config = FeaturizerConfig.model_validate(
        {"name": "tissue", "options": {"allow_missing": True}},
    )
    assert config.options is not None
    with pytest.raises(TypeError):
        config.options["allow_missing"] = False  # type: ignore[index]


def test_concat_children_are_tuple_not_list() -> None:
    register_builtin_components()
    config = CellLineFeaturizerConfig.model_validate(
        ["scaledGeneExpression", {"pca[methylation]": {"n_components": 32}}],
    )
    assert config.name == "concatFeaturizers"
    assert isinstance(config.featurizers, tuple)
    assert config.featurizers is not None
    assert config.featurizers[1].name == "pca"
    with pytest.raises(TypeError):
        config.featurizers[0] = CellLineFeaturizerConfig(name="raw", view="expression")  # type: ignore[index]


def test_model_config_replace_returns_new_validated_instance() -> None:
    register_builtin_components()
    original = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="scaledGeneExpression"),
        drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
        predictor=PredictorConfig(name="elasticNet"),
    )
    updated = original.replace(predictor=PredictorConfig(name="ridge"))
    assert updated.predictor.name == "ridge"
    assert original.predictor.name == "elasticNet"


def test_predictor_shorthand_writes_hyperparameter_space_defaults() -> None:
    register_builtin_components()
    config = ModelConfig.model_validate(
        {
            "cell_line_featurizer": "scaledGeneExpression",
            "drug_featurizer": "fingerprints",
            "predictor": {"randomForest": {"n_estimators": 10}},
        }
    )
    space = config.predictor.hyperparameter_space
    assert space is not None
    assert space["n_estimators"]["default"] == 10
    with pytest.raises(AttributeError):
        _ = config.predictor.hyperparameters  # type: ignore[attr-defined]


def test_zoo_config_copy_isolation() -> None:
    register_builtin_components()
    first = get_zoo_config("MultiViewLightGBM")
    second = get_zoo_config("MultiViewLightGBM")
    assert first == second
    assert first is not second
    assert first.cell_line_featurizer is not None
    child = first.cell_line_featurizer.featurizers[1]
    assert child.hyperparameter_space is not None
    with pytest.raises(TypeError):
        child.hyperparameter_space["n_components"] = {"default": 8}  # type: ignore[index]
