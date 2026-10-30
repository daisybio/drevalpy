"""Tests for the qualified hyperparameter keys of a model: resolving user input and checking keys."""

from __future__ import annotations

import pytest

from drevalpy.models.config import from_recipe
from drevalpy.models.config.model import ModelConfig
from drevalpy.models.hyperparameters import component_values, key, qualify, validate

TWO_PCA = "pca[expression]+pca[proteomics]:fingerprints:randomForest"


@pytest.fixture(scope="module")
def config() -> ModelConfig:
    return from_recipe(TWO_PCA)


def test_key_and_component_values_round_trip() -> None:
    qualified = key("sample_featurizer", "pca[methylation]", "n_components")

    assert qualified == "sample_featurizer.pca[methylation].n_components"
    values = {qualified: 4, "predictor.elasticNet.alpha": 1.0}
    assert component_values(values, "sample_featurizer", "pca[methylation]") == {"n_components": 4}
    assert component_values(values, "predictor", "elasticNet") == {"alpha": 1.0}
    assert component_values(values, "drug_featurizer", "fingerprints") == {}


def test_short_and_qualified_keys_resolve_to_the_same_mapping(config: ModelConfig) -> None:
    qualified = {
        "predictor.randomForest.n_estimators": 10,
        "sample_featurizer.pca[expression].n_components": 4,
    }
    assert qualify(config, qualified) == qualified
    assert qualify(config, {"n_estimators": 10, "sample_featurizer.pca[expression].n_components": 4}) == qualified
    assert qualify(config, {}) == {}


def test_ambiguous_unknown_and_duplicate_keys_raise(config: ModelConfig) -> None:
    with pytest.raises(ValueError, match="Ambiguous hyperparameter 'n_components'"):
        qualify(config, {"n_components": 4})
    with pytest.raises(ValueError, match="Unknown hyperparameter 'nope'"):
        qualify(config, {"nope": 1})
    with pytest.raises(ValueError, match="Duplicate hyperparameter assignment"):
        qualify(config, {"n_estimators": 1, "predictor.randomForest.n_estimators": 2})


def test_validate_accepts_only_qualified_keys_of_the_space(config: ModelConfig) -> None:
    validate(config, {})
    validate(config, {"predictor.randomForest.n_estimators": 10})


@pytest.mark.parametrize(
    "name",
    [
        "n_estimators",
        "predictor.randomForest.not_a_knob",
        "predictor.elasticNet.alpha",
        "sample_featurizer.pca.n_components",
        "drug_featurizer.fingerprints[smiles].radius",
        "sample_featurizer.concatFeaturizers.n_components",
    ],
    ids=["unqualified", "unknown-param", "another-predictor", "missing-view", "unexpected-view", "concat-parent"],
)
def test_validate_rejects_a_key_the_model_does_not_own(config: ModelConfig, name: str) -> None:
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        validate(config, {name: 1})
