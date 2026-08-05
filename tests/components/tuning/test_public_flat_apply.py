"""Tests for unified public flat hyperparameter application."""

from __future__ import annotations

import pytest

import drevalpy.components.register_builtins as register_builtins
from drevalpy.components.tuning.public_flat import apply_public_hyperparameters_to_config
from drevalpy.models.config import ModelConfig
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
    from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
    from drevalpy.models.config import FeaturizerConfig

    config = get_zoo_config("MultiViewRandomForest")
    updated = apply_public_hyperparameters_to_config(config, {"methylation_pca_components": 7})
    assert updated.cell_line_featurizer is not None
    children = updated.cell_line_featurizer.hyperparameters.get("featurizers", [])
    n_components: int | None = None
    for child in children:
        child_cfg = FeaturizerConfig.model_validate(
            normalize_featurizer_config(child, default_registry="cell_line"),
        )
        if child_cfg.name == "pca" and child_cfg.view == "methylation":
            n_components = int(child_cfg.hyperparameters["n_components"])
            break
    assert n_components == 7


def test_unknown_flat_keys_rejected() -> None:
    config = get_zoo_config("ElasticNet")
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        apply_public_hyperparameters_to_config(config, {"alpha": 0.1, "totally_unknown_key": 42})


def test_ambiguous_n_components_rejected_for_multi_pca_stack() -> None:
    config = ModelConfig.from_spec("pca[expression]+pca[proteomics]:fingerprints:randomForest")
    with pytest.raises(ValueError, match="Ambiguous hyperparameter 'n_components'"):
        apply_public_hyperparameters_to_config(config, {"n_components": 64})


def test_qualified_n_components_update_single_leaf() -> None:
    config = ModelConfig.from_spec("pca[expression]+pca[proteomics]:fingerprints:randomForest")
    updated = apply_public_hyperparameters_to_config(
        config,
        {
            "cell_line_featurizer.pca[expression].n_components": 32,
            "cell_line_featurizer.pca[proteomics].n_components": 16,
        },
    )
    from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
    from drevalpy.models.config import FeaturizerConfig

    assert updated.cell_line_featurizer is not None
    children = updated.cell_line_featurizer.hyperparameters["featurizers"]
    values: dict[str, int] = {}
    for child in children:
        child_cfg = FeaturizerConfig.model_validate(
            normalize_featurizer_config(child, default_registry="cell_line"),
        )
        if child_cfg.name == "pca":
            values[str(child_cfg.view)] = int(child_cfg.hyperparameters["n_components"])
    assert values == {"gene_expression": 32, "proteomics": 16}
