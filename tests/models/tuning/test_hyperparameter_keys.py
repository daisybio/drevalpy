"""Tests for hyperparameter keys."""

from __future__ import annotations

import pytest

from drevalpy.models.config import from_spec
from drevalpy.models.tuning.hyperparameter_keys import build_ownership_index, resolve_to_qualified_mapping


def _resolve(config, mapping):
    return resolve_to_qualified_mapping(config, mapping, build_ownership_index(config), reserved_keys=frozenset())


def test_single_owner_and_ambiguous_short_keys() -> None:
    index = build_ownership_index(from_spec("ElasticNet"))
    assert [t.qualified_key for t in index.short_to_targets["alpha"]] == ["predictor.elasticNet.alpha"]

    config = from_spec("pca[expression]+pca[proteomics]:fingerprints:randomForest")
    assert len(build_ownership_index(config).short_to_targets["n_components"]) == 2
    qualified = {
        "sample_featurizer.pca[expression].n_components": 32,
        "sample_featurizer.pca[proteomics].n_components": 16,
    }
    assert _resolve(config, qualified) == qualified
    with pytest.raises(ValueError, match="Ambiguous hyperparameter 'n_components'"):
        _resolve(config, {"n_components": 64})


def test_duplicate_short_and_qualified_assignments_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate hyperparameter assignment"):
        _resolve(from_spec("ElasticNet"), {"alpha": 0.2, "predictor.elasticNet.alpha": 0.3})
