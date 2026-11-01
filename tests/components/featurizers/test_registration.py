"""Tests for featurizer selectors, selector uniqueness and leaf kwargs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drevalpy.components.featurizers._featurizer_label import requires_explicit_view
from drevalpy.models.config import FeaturizerConfig, ModelConfig, PredictorConfig, from_recipe
from drevalpy.types.data.feature_source import Side


def test_selector_uses_view_brackets() -> None:
    assert FeaturizerConfig(name="pca", side="sample", view="gene_expression").selector == "pca[gene_expression]"
    assert FeaturizerConfig(name="view", side="sample", view="mutations").selector == "view[mutations]"
    assert FeaturizerConfig(name="landmarkGenes", side="sample").selector == "landmarkGenes"
    assert FeaturizerConfig(name="fingerprints", side="drug").selector == "fingerprints"


def test_explicit_view_requirement() -> None:
    assert requires_explicit_view("view", Side.SAMPLE) and requires_explicit_view("pca", Side.SAMPLE)
    assert not requires_explicit_view("landmarkGenes", Side.SAMPLE)


def test_unique_selectors_allow_same_name_different_views() -> None:
    config = FeaturizerConfig.parse(
        {
            "name": "concatFeaturizers",
            "featurizers": [
                {"name": "view", "view": "gene_expression"},
                {"name": "view", "view": "mutations"},
            ],
        },
        "sample",
    )
    assert [leaf.selector for leaf in config.leaves()] == ["view[gene_expression]", "view[mutations]"]


def test_featurizer_config_rejects_duplicate_qualified_selector() -> None:
    with pytest.raises(ValidationError, match="Duplicate featurizer selector 'view\\[gene_expression\\]'"):
        FeaturizerConfig.parse(
            {
                "name": "concatFeaturizers",
                "featurizers": [
                    {"name": "view", "view": "gene_expression"},
                    {"name": "view", "view": "gene_expression"},
                ],
            },
            "sample",
        )


def test_recipe_string_selector_uniqueness() -> None:
    with pytest.raises(ValueError, match="Duplicate featurizer selector"):
        from_recipe("view[expression]+view[expression]:fingerprints:randomForest")
    config = from_recipe("view[expression]+view[mutations]:fingerprints:randomForest")
    assert isinstance(config, ModelConfig)
    assert config.sample_featurizer is not None
    assert config.sample_featurizer.name == "concatFeaturizers"


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        sample_featurizer=FeaturizerConfig(name="pca", side="sample", view="gene_expression"),
        drug_featurizer=FeaturizerConfig(name="fingerprints", side="drug"),
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
            FeaturizerConfig.parse(
                {"name": "pca", "view": "gene_expression", "defaults": {"n_components": 16}}, "sample"
            ),
            "n_components",
            16,
        ),
        (
            FeaturizerConfig(
                name="pca",
                side="sample",
                view="gene_expression",
                hyperparameter_space={"n_components": {"type": "int", "low": 2, "high": 4, "default": 3}},
            ),
            "n_components",
            3,
        ),
        (
            FeaturizerConfig(name="view", side="sample", view="gene_expression", options={"view": "mutations"}),
            "view",
            "mutations",
        ),
    ],
    ids=["defaults-over-registry", "declared-space-over-registry", "explicit-view-option"],
)
def test_leaf_kwargs_precedence(leaf, key, expected) -> None:
    assert leaf.leaf_kwargs()[key] == expected
