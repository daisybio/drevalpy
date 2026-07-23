"""Tests for unified public flat hyperparameter application."""

from __future__ import annotations

import pytest

from drevalpy.models.flat_hyperparameters import apply_public_flat_hyperparameters
from drevalpy.models.zoo import get_zoo_config


def test_view_overrides_rewrite_featurizers() -> None:
    config = get_zoo_config("MultiViewRandomForest")
    updated = apply_public_flat_hyperparameters(
        config,
        {"cell_line_views": ["gene_expression"], "n_estimators": 12},
    )
    assert updated.cell_line_featurizer is not None
    assert updated.cell_line_featurizer.name == "scaledGeneExpression"
    assert "cell_line_views" not in updated.predictor.hyperparameters
    assert updated.predictor.hyperparameters["n_estimators"] == 12


def test_methylation_pca_components_alias() -> None:
    from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
    from drevalpy.models.config import FeaturizerConfig

    config = get_zoo_config("MultiViewRandomForest")
    updated = apply_public_flat_hyperparameters(config, {"methylation_pca_components": 7})
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
    with pytest.raises(ValueError, match="Unknown public hyperparameters"):
        apply_public_flat_hyperparameters(config, {"alpha": 0.1, "totally_unknown_key": 42})
