"""Public hyperparameter compatibility through ModelSpec."""

from __future__ import annotations

import pytest

from drevalpy.models import DRPModel, ModelSpec
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


@pytest.mark.parametrize("model_name", ["ElasticNet", "RandomForest", "NaiveMeanEffectsPredictor"])
def test_model_spec_defaults_build_without_error(model_name: str) -> None:
    spec = ModelSpec.from_name(model_name)
    model = DRPModel(spec, spec.default_hyperparameters)
    assert model.hyperparameters == spec.default_hyperparameters


def test_model_spec_defaults_have_no_namespaced_keys() -> None:
    spec = ModelSpec("PcaOneHotRF", from_spec("pca[expression]:identity:randomForest"))
    assert not any("." in key for key in spec.default_hyperparameters)


def test_public_round_trip_for_custom_spec() -> None:
    config = from_spec("pca[expression]:identity:randomForest")
    public = public_hyperparameters_from_config(config)
    rebuilt = apply_public_hyperparameters_to_config(config, public)
    assert rebuilt.featurizer_values("sample", "pca[expression]")["n_components"] == 128


def test_tuned_config_strips_structured_keys() -> None:
    config = from_spec("pca[expression]:identity:randomForest")
    merged = defaults_from_merged_space(merge_model_config_spaces(config))
    tuned = apply_merged_to_model_config(config, merged)
    public = public_hyperparameters_from_config(tuned)
    assert "sample_featurizer.pca[expression].n_components" not in public


def test_unknown_public_view_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        apply_public_hyperparameters_to_config(
            from_spec("MultiViewRandomForest"), {"sample_views": ["gene_expression"]}
        )


def test_methylation_pca_alias_round_trip() -> None:
    config = from_spec("MultiViewRandomForest")
    rebuilt = apply_public_hyperparameters_to_config(config, {"methylation_n_components": 9})
    assert rebuilt.featurizer_values("sample", "pca[methylation]")["n_components"] == 9


def test_cli_named_model_resolves_to_spec() -> None:
    assert ModelSpec.from_name("ElasticNet").name == "ElasticNet"
