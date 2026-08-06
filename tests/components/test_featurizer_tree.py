"""Tests for featurizer tree uniqueness helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drevalpy.components.featurizer_tree import ensure_unique_qualified_featurizers
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.models.config import FeaturizerConfig, from_spec


def test_ensure_unique_allows_same_name_different_views() -> None:
    config = FeaturizerConfig.model_validate(
        {
            "name": "concatFeaturizers",
            "registry": "cell_line",
            "hyperparameters": {
                "featurizers": [
                    {"name": "raw", "view": "gene_expression"},
                    {"name": "raw", "view": "mutations"},
                ],
            },
        },
    )
    ensure_unique_qualified_featurizers(config, "cell_line")


def test_featurizer_config_rejects_duplicate_qualified_selector() -> None:
    with pytest.raises(ValidationError, match="Duplicate featurizer selector 'raw\\[expression\\]'"):
        FeaturizerConfig.model_validate(
            {
                "name": "concatFeaturizers",
                "registry": "cell_line",
                "hyperparameters": {
                    "featurizers": [
                        {"name": "raw", "view": "gene_expression"},
                        {"name": "raw", "view": "gene_expression"},
                    ],
                },
            },
        )


def test_recipe_string_rejects_duplicate_qualified_selector() -> None:
    register_builtin_components()
    with pytest.raises(ValidationError, match="Duplicate featurizer selector"):
        from_spec("raw[expression]+raw[expression]:fingerprints:randomForest")


def test_recipe_string_allows_same_name_different_views() -> None:
    register_builtin_components()
    config = from_spec("raw[expression]+raw[mutations]:fingerprints:randomForest")
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.name == "concatFeaturizers"
