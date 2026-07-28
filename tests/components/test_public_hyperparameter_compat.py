"""Public hyperparameter compatibility tests against development-style usage."""

from __future__ import annotations

import pytest

import drevalpy.components.register_builtins as register_builtins
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
    assert "featurizer.cell_line.pca[expression].n_components" not in defaults
    assert "featurizer.cell_line.pca.0.n_components" not in defaults
    model_cls(defaults)


def test_default_config_has_component_local_hyperparameters_only() -> None:
    model_cls = construct_model("PcaOneHotRF", "pca[expression]:identity:randomForest")
    config = default_config_for_drp_model(model_cls)
    assert config is not None
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.hyperparameters == {"n_components": 128}
    assert_component_local_hyperparameters(config)


def test_public_round_trip_for_constructed_model() -> None:
    model_cls = construct_model("PcaOneHotRF", "pca[expression]:identity:randomForest")
    config = default_config_for_drp_model(model_cls)
    assert config is not None
    public = public_hyperparameters_from_config(config)
    rebuilt = config_from_public_hyperparameters(model_cls, public)
    assert rebuilt is not None
    assert rebuilt.cell_line_featurizer is not None
    assert rebuilt.cell_line_featurizer.hyperparameters == {"n_components": 128}
    assert_component_local_hyperparameters(rebuilt)


def test_tuned_config_strips_structured_keys() -> None:
    model_cls = construct_model("PcaOneHotRF", "pca[expression]:identity:randomForest")
    base = default_config_for_drp_model(model_cls)
    assert base is not None
    merged = defaults_from_merged_space(merge_model_config_spaces(base))
    tuned = tuned_config_for_drp_model(model_cls, merged)
    assert tuned is not None
    assert_component_local_hyperparameters(tuned)
    public = public_hyperparameters_from_config(tuned)
    assert "featurizer.cell_line.pca[expression].n_components" not in public
    assert "featurizer.cell_line.pca.0.n_components" not in public


def test_apply_merged_never_leaks_namespaced_keys_into_components() -> None:
    from drevalpy.models.config import ModelConfig

    config = ModelConfig.from_spec("pca[expression]:identity:randomForest")
    merged = defaults_from_merged_space(merge_model_config_spaces(config))
    updated = apply_merged_to_model_config(config, merged)
    assert_component_local_hyperparameters(updated)


def test_pca_methylation_pca_components_alias_round_trip() -> None:
    rebuilt = config_from_public_hyperparameters(
        construct_model("MultiViewRandomForest"),
        {"methylation_pca_components": 9},
    )
    assert rebuilt is not None
    assert rebuilt.cell_line_featurizer is not None
    children = rebuilt.cell_line_featurizer.hyperparameters["featurizers"]
    pca_child = next(child for child in children if child["name"] == "pca")
    assert pca_child["hyperparameters"]["n_components"] == 9


def test_cell_line_views_override_on_configure_path() -> None:
    rebuilt = config_from_public_hyperparameters(
        construct_model("MultiViewRandomForest"),
        {"cell_line_views": ["gene_expression"]},
    )
    assert rebuilt is not None
    assert rebuilt.cell_line_featurizer is not None
    assert rebuilt.cell_line_featurizer.name == "scaledGeneExpression"


def test_pca_methylation_flat_key_round_trip() -> None:
    from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
    from drevalpy.models.config import FeaturizerConfig, ModelConfig, PredictorConfig

    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig.model_validate(
            normalize_featurizer_config(
                [
                    "scaledGeneExpression",
                    {"pca[methylation]": {"n_components": 100}},
                ],
                default_registry="cell_line",
            ),
        ),
        drug_featurizer=FeaturizerConfig.model_validate(
            normalize_featurizer_config("fingerprints", default_registry="drug"),
        ),
        predictor=PredictorConfig(name="randomForest"),
    )
    public = public_hyperparameters_from_config(config)
    assert public["methylation_n_components"] == 100
    rebuilt = config_from_public_hyperparameters(construct_model("MultiViewRandomForest"), public)
    assert rebuilt is not None
    assert rebuilt.cell_line_featurizer is not None
    children = rebuilt.cell_line_featurizer.hyperparameters["featurizers"]
    pca_child = next(child for child in children if child["name"] == "pca")
    assert pca_child["view"] == "methylation"
    assert pca_child["hyperparameters"]["n_components"] == 100


def test_cli_resolves_models_through_construct_model() -> None:
    from drevalpy.cli.run_cv import run_hpam_split

    model_class = construct_model("ElasticNet")
    assert model_class.get_model_name() == "ElasticNet"
    assert callable(run_hpam_split)
