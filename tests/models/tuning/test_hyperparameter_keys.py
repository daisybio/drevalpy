"""Tests for hyperparameter ownership indexing and resolution."""

from __future__ import annotations

import pytest

from drevalpy.models import construct_model
from drevalpy.models.config import from_spec
from drevalpy.models.config.model import ModelConfig
from drevalpy.models.tuning.hyperparameter_keys import (
    build_ownership_index,
    resolve_to_qualified_mapping,
)
from drevalpy.registry._builtins import register_builtin_components


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def test_elastic_net_alpha_has_single_owner() -> None:
    model_cls = construct_model("ElasticNet")
    from drevalpy.models.tuning.config_resolution import default_config_for_drp_model

    config = default_config_for_drp_model(model_cls)
    assert config is not None
    index = build_ownership_index(config.template)
    assert len(index.short_to_targets["alpha"]) == 1
    assert index.short_to_targets["alpha"][0].qualified_key == "predictor.elasticNet.alpha"


def test_two_pca_views_make_n_components_ambiguous() -> None:
    config = from_spec("pca[expression]+pca[proteomics]:fingerprints:randomForest")
    assert isinstance(config, ModelConfig)
    index = build_ownership_index(config)
    owners = index.short_to_targets["n_components"]
    assert len(owners) == 2
    qualified = resolve_to_qualified_mapping(
        config,
        {
            "cell_line_featurizer.pca[expression].n_components": 32,
            "cell_line_featurizer.pca[proteomics].n_components": 16,
        },
        index,
        reserved_keys=frozenset(),
    )
    assert qualified["cell_line_featurizer.pca[expression].n_components"] == 32
    assert qualified["cell_line_featurizer.pca[proteomics].n_components"] == 16


def test_ambiguous_short_key_lists_qualified_alternatives() -> None:
    config = from_spec("pca[expression]+pca[proteomics]:fingerprints:randomForest")
    assert isinstance(config, ModelConfig)
    index = build_ownership_index(config)
    with pytest.raises(ValueError, match="Ambiguous hyperparameter 'n_components'"):
        resolve_to_qualified_mapping(config, {"n_components": 64}, index, reserved_keys=frozenset())


def test_duplicate_short_and_qualified_assignments_rejected() -> None:
    model_cls = construct_model("ElasticNet")
    from drevalpy.models.tuning.config_resolution import default_config_for_drp_model

    config = default_config_for_drp_model(model_cls)
    assert config is not None
    index = build_ownership_index(config.template)
    with pytest.raises(ValueError, match="Duplicate hyperparameter assignment"):
        resolve_to_qualified_mapping(
            config.template,
            {"alpha": 0.2, "predictor.elasticNet.alpha": 0.3},
            index,
            reserved_keys=frozenset(),
        )
