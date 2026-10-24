"""Tests for featurizer leaf kwarg resolution (``_leaf_kwargs``)."""

from __future__ import annotations

import pytest

from drevalpy.components.featurizers._leaf_kwargs import featurizer_leaf_kwargs
from drevalpy.models.config import (
    DrugFeaturizerConfig,
    ModelConfig,
    PredictorConfig,
    SampleFeaturizerConfig,
)
from drevalpy.models.config.resolved import ResolvedModelConfig


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        sample_featurizer=SampleFeaturizerConfig(name="pca", view="gene_expression"),
        drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
        predictor=PredictorConfig(name="elasticNet"),
    )


def test_leaf_kwargs_fill_hyperparameter_defaults_from_the_registry(model_config: ModelConfig) -> None:
    sample = featurizer_leaf_kwargs(model_config.sample_featurizer, registry="sample", resolved=None)
    drug = featurizer_leaf_kwargs(model_config.drug_featurizer, registry="drug", resolved=None)

    assert sample == {"n_components": 128, "view": "gene_expression"}
    assert drug == {"radius": 2, "n_bits": 2048, "use_chirality": False, "use_counts": False}


@pytest.mark.parametrize(
    ("leaf", "key", "expected"),
    [
        (SampleFeaturizerConfig(name="pca", view="gene_expression", options={"n_components": 4}), "n_components", 4),
        (
            SampleFeaturizerConfig(
                name="pca",
                view="gene_expression",
                hyperparameter_space={"n_components": {"type": "int", "low": 2, "high": 4, "default": 3}},
            ),
            "n_components",
            3,
        ),
        (
            SampleFeaturizerConfig(name="raw", view="gene_expression", options={"view": "mutations"}),
            "view",
            "mutations",
        ),
    ],
    ids=["options-over-defaults", "declared-space-over-registry", "explicit-view-option"],
)
def test_leaf_kwargs_precedence(leaf, key, expected) -> None:
    assert featurizer_leaf_kwargs(leaf, registry="sample", resolved=None)[key] == expected


def test_leaf_kwargs_ignore_space_entries_without_a_default() -> None:
    # ``FeaturizerConfig`` rejects such an entry, so only a config-shaped stub reaches the skip branch.
    class _LeafWithoutDefault:
        name = "pca"
        view = "gene_expression"
        options = None
        hyperparameter_space = {"n_components": {"type": "int", "low": 2, "high": 4}}

    kwargs = featurizer_leaf_kwargs(_LeafWithoutDefault(), registry="sample", resolved=None)

    assert kwargs == {"view": "gene_expression"}


def test_leaf_kwargs_let_resolved_values_win(model_config: ModelConfig) -> None:
    resolved = ResolvedModelConfig(
        template=model_config,
        values={"sample_featurizer.pca[gene_expression].n_components": 16},
    )

    kwargs = featurizer_leaf_kwargs(model_config.sample_featurizer, registry="sample", resolved=resolved)

    assert kwargs["n_components"] == 16
