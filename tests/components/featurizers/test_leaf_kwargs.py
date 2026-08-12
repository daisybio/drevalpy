"""Tests for featurizer leaf kwarg resolution.

Mirrors :mod:`drevalpy.components.featurizers._leaf_kwargs`, whose only caller is
``drevalpy.models.config.view_resolution.views_from_featurizer_config``.
"""

from __future__ import annotations

import pytest

from drevalpy.components.featurizers._leaf_kwargs import featurizer_leaf_kwargs
from drevalpy.models.config import (
    CellLineFeaturizerConfig,
    DrugFeaturizerConfig,
    ModelConfig,
    PredictorConfig,
)
from drevalpy.models.config.resolved import ResolvedModelConfig


@pytest.fixture
def model_config() -> ModelConfig:
    """A pca[gene_expression] + fingerprints + elasticNet stack."""
    return ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="pca", view="gene_expression"),
        drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
        predictor=PredictorConfig(name="elasticNet"),
    )


def test_leaf_kwargs_fill_hyperparameter_defaults_from_the_registry(model_config: ModelConfig) -> None:
    leaf = model_config.cell_line_featurizer
    assert leaf is not None

    kwargs = featurizer_leaf_kwargs(leaf, registry="cell_line", resolved=None)

    assert kwargs == {"n_components": 128, "view": "gene_expression"}


def test_leaf_kwargs_fill_defaults_for_drug_featurizers(model_config: ModelConfig) -> None:
    leaf = model_config.drug_featurizer
    assert leaf is not None

    kwargs = featurizer_leaf_kwargs(leaf, registry="drug", resolved=None)

    assert kwargs == {"radius": 2, "n_bits": 2048, "use_chirality": False, "use_counts": False}


def test_leaf_kwargs_prefer_config_options_over_registry_defaults() -> None:
    leaf = CellLineFeaturizerConfig(name="pca", view="gene_expression", options={"n_components": 4})

    kwargs = featurizer_leaf_kwargs(leaf, registry="cell_line", resolved=None)

    assert kwargs["n_components"] == 4


def test_leaf_kwargs_prefer_a_declared_space_over_the_registry_space() -> None:
    leaf = CellLineFeaturizerConfig(
        name="pca",
        view="gene_expression",
        hyperparameter_space={"n_components": {"type": "int", "low": 2, "high": 4, "default": 3}},
    )

    kwargs = featurizer_leaf_kwargs(leaf, registry="cell_line", resolved=None)

    assert kwargs["n_components"] == 3


def test_leaf_kwargs_ignore_space_entries_without_a_default() -> None:
    # ``FeaturizerConfig`` rejects a space entry without a ``default``, so the
    # skip branch is only reachable through a config-shaped stub.
    class _LeafWithoutDefault:
        name = "pca"
        view = "gene_expression"
        options = None
        hyperparameter_space = {"n_components": {"type": "int", "low": 2, "high": 4}}

    kwargs = featurizer_leaf_kwargs(_LeafWithoutDefault(), registry="cell_line", resolved=None)

    assert kwargs == {"view": "gene_expression"}


def test_leaf_kwargs_let_resolved_values_win(model_config: ModelConfig) -> None:
    leaf = model_config.cell_line_featurizer
    assert leaf is not None
    resolved = ResolvedModelConfig(
        template=model_config,
        values={"cell_line_featurizer.pca[gene_expression].n_components": 16},
    )

    kwargs = featurizer_leaf_kwargs(leaf, registry="cell_line", resolved=resolved)

    assert kwargs["n_components"] == 16


def test_leaf_kwargs_omit_view_when_the_config_declares_none(model_config: ModelConfig) -> None:
    leaf = model_config.drug_featurizer
    assert leaf is not None

    kwargs = featurizer_leaf_kwargs(leaf, registry="drug", resolved=None)

    assert "view" not in kwargs


def test_leaf_kwargs_do_not_override_an_explicit_view_option() -> None:
    leaf = CellLineFeaturizerConfig(name="raw", view="gene_expression", options={"view": "mutations"})

    kwargs = featurizer_leaf_kwargs(leaf, registry="cell_line", resolved=None)

    assert kwargs["view"] == "mutations"
