"""Public hyperparameter compatibility tests against development-style usage."""

from __future__ import annotations

import drevalpy.components.register_builtins as register_builtins
import pytest

from drevalpy.components.tuning.drp_hyperparameters import (
    assert_component_local_hyperparameters,
    config_from_public_hyperparameters,
    default_config_for_drp_model,
    public_hyperparameters_from_config,
    tuned_config_for_drp_model,
)
from drevalpy.components.tuning.search_space import (
    apply_merged_to_model_config,
    defaults_from_merged_space,
    merge_model_config_spaces,
)
from drevalpy.models import MODEL_FACTORY, construct_model


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtins.register_builtin_components()


@pytest.mark.parametrize(
    "model_name",
    ["ElasticNet", "RandomForest", "NaiveMeanEffectsPredictor"],
)
def test_model_factory_defaults_build_without_error(model_name: str) -> None:
    model_cls = MODEL_FACTORY[model_name]
    model = model_cls()
    defaults = model_cls.get_default_hyperparameters()
    model.build_model(defaults)
    assert isinstance(defaults, dict)


def test_construct_model_defaults_have_no_namespaced_keys() -> None:
    model_cls = construct_model("PcaOneHotRF", "pca:identity:randomForest")
    defaults = model_cls.get_default_hyperparameters()
    assert not any("." in key for key in defaults)
    assert "featurizer.cell_line.pca.0.n_components" not in defaults
    model_cls().build_model(defaults)


def test_default_config_has_component_local_hyperparameters_only() -> None:
    model_cls = construct_model("PcaOneHotRF", "pca:identity:randomForest")
    config = default_config_for_drp_model(model_cls)
    assert config is not None
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.hyperparameters == {"n_components": 128}
    assert_component_local_hyperparameters(config)


def test_public_round_trip_for_constructed_model() -> None:
    model_cls = construct_model("PcaOneHotRF", "pca:identity:randomForest")
    config = default_config_for_drp_model(model_cls)
    assert config is not None
    public = public_hyperparameters_from_config(config)
    rebuilt = config_from_public_hyperparameters(model_cls, public)
    assert rebuilt is not None
    assert rebuilt.cell_line_featurizer is not None
    assert rebuilt.cell_line_featurizer.hyperparameters == {"n_components": 128}
    assert_component_local_hyperparameters(rebuilt)


def test_tuned_config_strips_structured_keys() -> None:
    model_cls = construct_model("PcaOneHotRF", "pca:identity:randomForest")
    base = default_config_for_drp_model(model_cls)
    assert base is not None
    merged = defaults_from_merged_space(merge_model_config_spaces(base))
    tuned = tuned_config_for_drp_model(model_cls, merged)
    assert tuned is not None
    assert_component_local_hyperparameters(tuned)
    public = public_hyperparameters_from_config(tuned)
    assert "featurizer.cell_line.pca.0.n_components" not in public


def test_apply_merged_never_leaks_namespaced_keys_into_components() -> None:
    from drevalpy.models.config import ModelConfig

    config = ModelConfig.from_spec("pca:identity:randomForest")
    merged = defaults_from_merged_space(merge_model_config_spaces(config))
    updated = apply_merged_to_model_config(config, merged)
    assert_component_local_hyperparameters(updated)


def test_cli_resolves_models_through_model_factory() -> None:
    from drevalpy.cli_run_cv import run_hpam_split

    model_class = MODEL_FACTORY["ElasticNet"]
    assert model_class.get_model_name() == "ElasticNet"
    assert callable(run_hpam_split)
