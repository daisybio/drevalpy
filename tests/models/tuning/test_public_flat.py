"""Tests for public flat."""

from __future__ import annotations

import pytest

from drevalpy.models import ModelSpec
from drevalpy.models.config import from_spec
from drevalpy.models.tuning.public_flat import (
    apply_public_hyperparameters_to_config,
    public_hyperparameters_from_config,
)
from drevalpy.models.tuning.search_space import (
    apply_merged_to_model_config,
    defaults_from_merged_space,
    merge_model_config_spaces,
)
from drevalpy.models.zoo import get_zoo_config

TWO_PCA = "pca[expression]+pca[proteomics]:fingerprints:randomForest"
QUALIFIED = {
    "sample_featurizer.pca[expression].n_components": 32,
    "sample_featurizer.pca[proteomics].n_components": 16,
}


def test_public_round_trip_restores_defaults() -> None:
    config = from_spec("pca[expression]:identity:randomForest")
    public = public_hyperparameters_from_config(config)
    assert not any("." in key for key in public)
    rebuilt = apply_public_hyperparameters_to_config(config, public)
    assert rebuilt.featurizer_values("sample", "pca[expression]")["n_components"] == 128

    elastic = get_zoo_config("ElasticNet")
    assert apply_public_hyperparameters_to_config(elastic, public_hyperparameters_from_config(elastic)) is not None


def test_methylation_n_components_alias() -> None:
    config = from_spec("scaledGeneExpression+pca[methylation]:fingerprints:randomForest")
    updated = apply_public_hyperparameters_to_config(config, {"methylation_n_components": 7})
    assert updated.featurizer_values("sample", "pca[methylation]")["n_components"] == 7


def test_qualified_n_components_update_single_leaf() -> None:
    updated = apply_public_hyperparameters_to_config(from_spec(TWO_PCA), QUALIFIED)
    assert updated.featurizer_values("sample", "pca[expression]")["n_components"] == 32
    assert updated.featurizer_values("sample", "pca[proteomics]")["n_components"] == 16


def test_ambiguous_n_components_rejected_for_multi_pca_stack() -> None:
    with pytest.raises(ValueError, match="Ambiguous hyperparameter 'n_components'"):
        apply_public_hyperparameters_to_config(from_spec(TWO_PCA), {"n_components": 64})


@pytest.mark.parametrize(
    ("model", "flat"),
    [("ElasticNet", {"alpha": 0.1, "totally_unknown_key": 42}), ("MultiViewRandomForest", {"sample_views": ["x"]})],
)
def test_unknown_flat_keys_rejected(model: str, flat: dict) -> None:
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        apply_public_hyperparameters_to_config(from_spec(model), flat)


def test_tuned_config_strips_structured_keys() -> None:
    config = from_spec("pca[expression]:identity:randomForest")
    tuned = apply_merged_to_model_config(config, defaults_from_merged_space(merge_model_config_spaces(config)))
    assert "sample_featurizer.pca[expression].n_components" not in public_hyperparameters_from_config(tuned)


def test_model_spec_default_hyperparameters_are_flat() -> None:
    spec = ModelSpec("PcaOneHotRF", from_spec("pca[expression]:identity:randomForest"))
    assert not any("." in key for key in spec.default_hyperparameters)
