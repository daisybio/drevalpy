"""Tests for public hyperparameter mapping export."""

from __future__ import annotations

import pytest

from drevalpy.models.config import from_spec
from drevalpy.models.config.model import ModelConfig
from drevalpy.models.tuning.hyperparameter_export import (
    export_public_mapping,
)
from drevalpy.registry._builtins import register_builtin_components


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def test_export_uses_qualified_keys_for_collisions() -> None:
    config = from_spec("pca[expression]+pca[proteomics]:fingerprints:randomForest")
    assert isinstance(config, ModelConfig)
    exported = export_public_mapping(config)
    assert "n_components" not in exported
    assert "cell_line_featurizer.pca[expression].n_components" in exported
    assert "cell_line_featurizer.pca[proteomics].n_components" in exported
