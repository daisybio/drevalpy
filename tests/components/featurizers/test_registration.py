"""Tests for featurizer selectors, selector uniqueness and leaf kwargs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drevalpy.components.featurizers._featurizer_label import (
    featurizer_config_block_label,
    qualified_featurizer_selector,
    requires_explicit_view,
)
from drevalpy.components.featurizers._featurizer_tree import ensure_unique_qualified_featurizers
from drevalpy.components.featurizers._leaf_kwargs import featurizer_leaf_kwargs
from drevalpy.models.config import FeaturizerConfig, ModelConfig, PredictorConfig, from_recipe


def test_qualified_selector_uses_view_brackets() -> None:
    assert qualified_featurizer_selector("pca", "gene_expression") == "pca[gene_expression]"
    assert qualified_featurizer_selector("raw", "mutations") == "raw[mutations]"
    assert qualified_featurizer_selector("landmarkGenes") == "landmarkGenes"


def test_block_labels_and_explicit_view_requirement() -> None:
    assert featurizer_config_block_label("pca", "proteomics") == "pca[proteomics]"
    assert featurizer_config_block_label("fingerprints", None) == "fingerprints"
    assert requires_explicit_view("raw") and requires_explicit_view("pca")
    assert not requires_explicit_view("landmarkGenes")


def test_ensure_unique_allows_same_name_different_views() -> None:
    config = FeaturizerConfig.model_validate(
        {
            "name": "concatFeaturizers",
            "featurizers": [
                {"name": "raw", "view": "gene_expression"},
                {"name": "raw", "view": "mutations"},
            ],
        },
    )
    ensure_unique_qualified_featurizers(config)


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


def test_ensure_unique_ignores_a_non_concat_node() -> None:
    config = FeaturizerConfig.model_validate({"name": "raw", "view": "gene_expression"})

    ensure_unique_qualified_featurizers(config)


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
    sample = featurizer_leaf_kwargs(model_config.sample_featurizer)
    drug = featurizer_leaf_kwargs(model_config.drug_featurizer)

    assert sample == {"n_components": 128, "view": "gene_expression"}
    assert drug == {"radius": 2, "n_bits": 2048, "use_chirality": False, "use_counts": False}


@pytest.mark.parametrize(
    ("leaf", "key", "expected"),
    [
        (FeaturizerConfig(name="pca", view="gene_expression", options={"n_components": 4}), "n_components", 4),
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
    ids=["options-over-defaults", "declared-space-over-registry", "explicit-view-option"],
)
def test_leaf_kwargs_precedence(leaf, key, expected) -> None:
    assert featurizer_leaf_kwargs(leaf)[key] == expected


def test_leaf_kwargs_ignore_space_entries_without_a_default() -> None:
    # ``FeaturizerConfig`` rejects such an entry, so only a config-shaped stub reaches the skip branch.
    class _LeafWithoutDefault:
        name = "pca"
        view = "gene_expression"
        options = None
        hyperparameter_space = {"n_components": {"type": "int", "low": 2, "high": 4}}

    kwargs = featurizer_leaf_kwargs(_LeafWithoutDefault())

    assert kwargs == {"view": "gene_expression"}
