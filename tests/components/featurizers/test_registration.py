"""Tests for featurizer selectors, selector uniqueness and leaf kwargs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drevalpy.components.featurizers._featurizer_label import requires_explicit_view
from drevalpy.models.config import FeaturizerConfig, ModelConfig, PredictorConfig, from_recipe


def test_selector_uses_view_brackets() -> None:
    assert FeaturizerConfig(name="pca", view="gene_expression").selector == "pca[gene_expression]"
    assert FeaturizerConfig(name="raw", view="mutations").selector == "raw[mutations]"
    assert FeaturizerConfig(name="landmarkGenes").selector == "landmarkGenes"
    assert FeaturizerConfig(name="fingerprints").selector == "fingerprints"


def test_explicit_view_requirement() -> None:
    assert requires_explicit_view("raw") and requires_explicit_view("pca")
    assert not requires_explicit_view("landmarkGenes")


def test_unique_selectors_allow_same_name_different_views() -> None:
    config = FeaturizerConfig.model_validate(
        {
            "name": "concatFeaturizers",
            "featurizers": [
                {"name": "raw", "view": "gene_expression"},
                {"name": "raw", "view": "mutations"},
            ],
        },
    )
    assert [leaf.selector for leaf in config.leaves()] == ["raw[gene_expression]", "raw[mutations]"]


def test_featurizer_config_rejects_duplicate_qualified_selector() -> None:
    with pytest.raises(ValidationError, match="Duplicate featurizer selector 'raw\\[gene_expression\\]'"):
        FeaturizerConfig.model_validate(
            {
                "name": "concatFeaturizers",
                "featurizers": [
                    {"name": "raw", "view": "gene_expression"},
                    {"name": "raw", "view": "gene_expression"},
                ],
            },
        )


def test_recipe_string_selector_uniqueness() -> None:
    with pytest.raises(ValueError, match="Duplicate featurizer selector"):
        from_recipe("raw[expression]+raw[expression]:fingerprints:randomForest")
    config = from_recipe("raw[expression]+raw[mutations]:fingerprints:randomForest")
    assert isinstance(config, ModelConfig)
    assert config.sample_featurizer is not None
    assert config.sample_featurizer.name == "concatFeaturizers"


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        sample_featurizer=FeaturizerConfig(name="pca", view="gene_expression"),
        drug_featurizer=FeaturizerConfig(name="fingerprints"),
        predictor=PredictorConfig(name="elasticNet"),
    )


def test_leaf_kwargs_fill_hyperparameter_defaults_from_the_registry(model_config: ModelConfig) -> None:
    sample = model_config.sample_featurizer.leaf_kwargs()
    drug = model_config.drug_featurizer.leaf_kwargs()

    assert sample == {"n_components": 128, "view": "gene_expression"}
    assert drug == {"radius": 2, "n_bits": 2048, "use_chirality": False, "use_counts": False}


@pytest.mark.parametrize(
    ("leaf", "key", "expected"),
    [
        (
            FeaturizerConfig.model_validate(
                {"name": "pca", "view": "gene_expression", "defaults": {"n_components": 16}}
            ),
            "n_components",
            16,
        ),
        (
            FeaturizerConfig(
                name="pca",
                view="gene_expression",
                hyperparameter_space={"n_components": {"type": "int", "low": 2, "high": 4, "default": 3}},
            ),
            "n_components",
            3,
        ),
        (
            FeaturizerConfig(name="raw", view="gene_expression", options={"view": "mutations"}),
            "view",
            "mutations",
        ),
    ],
    ids=["defaults-over-registry", "declared-space-over-registry", "explicit-view-option"],
)
def test_leaf_kwargs_precedence(leaf, key, expected) -> None:
    assert leaf.leaf_kwargs()[key] == expected
