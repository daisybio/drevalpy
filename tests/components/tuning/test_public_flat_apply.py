"""Tests for unified public flat hyperparameter application."""

from __future__ import annotations

import pytest

from drevalpy.components.registry import register_builtins
from drevalpy.models.config import from_spec
from drevalpy.models.config.model import ModelConfig
from drevalpy.models.tuning.public_flat import apply_public_hyperparameters_to_config
from drevalpy.models.zoo import get_zoo_config


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtins.register_builtin_components()


def test_view_keys_rejected() -> None:
    config = get_zoo_config("MultiViewRandomForest")
    with pytest.raises(ValueError, match=r"Legacy view keys|no longer supported"):
        apply_public_hyperparameters_to_config(
            config,
            {"cell_line_views": ["gene_expression"], "n_estimators": 12},
        )


def test_methylation_pca_components_alias() -> None:
    config = get_zoo_config("MultiViewRandomForest")
    updated = apply_public_hyperparameters_to_config(config, {"methylation_pca_components": 7})
    assert updated.featurizer_values("cell_line", "pca[methylation]")["n_components"] == 7


def test_unknown_flat_keys_rejected() -> None:
    config = get_zoo_config("ElasticNet")
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        apply_public_hyperparameters_to_config(config, {"alpha": 0.1, "totally_unknown_key": 42})


def test_ambiguous_n_components_rejected_for_multi_pca_stack() -> None:
    config = from_spec("pca[expression]+pca[proteomics]:fingerprints:randomForest")
    assert isinstance(config, ModelConfig)
    with pytest.raises(ValueError, match="Ambiguous hyperparameter 'n_components'"):
        apply_public_hyperparameters_to_config(config, {"n_components": 64})


def test_qualified_n_components_update_single_leaf() -> None:
    config = from_spec("pca[expression]+pca[proteomics]:fingerprints:randomForest")
    assert isinstance(config, ModelConfig)
    updated = apply_public_hyperparameters_to_config(
        config,
        {
            "cell_line_featurizer.pca[expression].n_components": 32,
            "cell_line_featurizer.pca[proteomics].n_components": 16,
        },
    )
    assert updated.featurizer_values("cell_line", "pca[expression]")["n_components"] == 32
    assert updated.featurizer_values("cell_line", "pca[proteomics]")["n_components"] == 16
