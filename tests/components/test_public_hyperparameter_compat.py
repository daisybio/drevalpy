"""Public hyperparameter compatibility tests against development-style usage."""

from __future__ import annotations

import pytest

import drevalpy.components.core.plugins.register_builtins as register_builtins
from drevalpy.components.core.tuning.drp_hyperparameters import (
    assert_component_local_hyperparameters,
    config_from_public_hyperparameters,
    default_config_for_drp_model,
    public_hyperparameters_from_config,
    tuned_config_for_drp_model,
)
from drevalpy.components.core.tuning.search_space import (
    apply_merged_to_model_config,
    defaults_from_merged_space,
    merge_model_config_spaces,
)
from drevalpy.models import construct_model


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtins.register_builtin_components()


@pytest.mark.parametrize(
    "model_name",
    ["ElasticNet", "RandomForest", "NaiveMeanEffectsPredictor"],
)
def test_model_factory_defaults_build_without_error(model_name: str) -> None:
    model_cls = construct_model(model_name)
    defaults = model_cls.get_default_hyperparameters()
    model = model_cls(defaults)
    assert isinstance(defaults, dict)
    assert model.hyperparameters == defaults


def test_construct_model_defaults_have_no_namespaced_keys() -> None:
    model_cls = construct_model("PcaOneHotRF", "pca[expression]:identity:randomForest")
    defaults = model_cls.get_default_hyperparameters()
    assert not any("." in key for key in defaults)
    assert "cell_line_featurizer.pca[expression].n_components" not in defaults
    assert "cell_line_featurizer.pca.0.n_components" not in defaults
    model_cls(defaults)


def test_default_config_has_component_local_hyperparameters_only() -> None:
    model_cls = construct_model("PcaOneHotRF", "pca[expression]:identity:randomForest")
    config = default_config_for_drp_model(model_cls)
    assert config is not None
    assert config.featurizer_values("cell_line", "pca[expression]")["n_components"] == 128
    assert_component_local_hyperparameters(config)


def test_public_round_trip_for_constructed_model() -> None:
    model_cls = construct_model("PcaOneHotRF", "pca[expression]:identity:randomForest")
    config = default_config_for_drp_model(model_cls)
    assert config is not None
    public = public_hyperparameters_from_config(config)
    rebuilt = config_from_public_hyperparameters(model_cls, public)
    assert rebuilt is not None
    assert rebuilt.featurizer_values("cell_line", "pca[expression]")["n_components"] == 128
    assert_component_local_hyperparameters(rebuilt)


def test_tuned_config_strips_structured_keys() -> None:
    model_cls = construct_model("PcaOneHotRF", "pca[expression]:identity:randomForest")
    base = default_config_for_drp_model(model_cls)
    assert base is not None
    merged = defaults_from_merged_space(merge_model_config_spaces(base.template))
    tuned = tuned_config_for_drp_model(model_cls, merged)
    assert tuned is not None
    assert_component_local_hyperparameters(tuned)
    public = public_hyperparameters_from_config(tuned)
    assert "cell_line_featurizer.pca[expression].n_components" not in public
    assert "cell_line_featurizer.pca.0.n_components" not in public


def test_apply_merged_never_leaks_namespaced_keys_into_components() -> None:
    from drevalpy.models.config import from_spec
    from drevalpy.models.config.model import ModelConfig

    config = from_spec("pca[expression]:identity:randomForest")
    assert isinstance(config, ModelConfig)
    merged = defaults_from_merged_space(merge_model_config_spaces(config))
    updated = apply_merged_to_model_config(config, merged)
    assert_component_local_hyperparameters(updated)


def test_pca_methylation_pca_components_alias_round_trip() -> None:
    rebuilt = config_from_public_hyperparameters(
        construct_model("MultiViewRandomForest"),
        {"methylation_pca_components": 9},
    )
    assert rebuilt is not None
    assert rebuilt.featurizer_values("cell_line", "pca[methylation]")["n_components"] == 9


def test_cell_line_views_override_on_configure_path_rejected() -> None:
    with pytest.raises(ValueError, match=r"Legacy view keys|no longer supported"):
        config_from_public_hyperparameters(
            construct_model("MultiViewRandomForest"),
            {"cell_line_views": ["gene_expression"]},
        )


def test_pca_methylation_flat_key_round_trip() -> None:
    from drevalpy.components.core.tuning.search_space import resolve_model_config
    from drevalpy.models.config import CellLineFeaturizerConfig, DrugFeaturizerConfig, ModelConfig, PredictorConfig

    template = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig.model_validate(
            [
                "scaledGeneExpression",
                {"pca[methylation]": {"n_components": 100}},
            ],
        ),
        drug_featurizer=DrugFeaturizerConfig.model_validate("fingerprints"),
        predictor=PredictorConfig(name="randomForest"),
    )
    config = resolve_model_config(
        template,
        {"cell_line_featurizer.pca[methylation].n_components": 100},
    )
    public = public_hyperparameters_from_config(config)
    assert public["n_components"] == 100
    rebuilt = config_from_public_hyperparameters(construct_model("MultiViewRandomForest"), public)
    assert rebuilt is not None
    assert rebuilt.featurizer_values("cell_line", "pca[methylation]")["n_components"] == 100


def test_cli_resolves_models_through_construct_model() -> None:
    model_class = construct_model("ElasticNet")
    assert model_class.get_model_name() == "ElasticNet"
