"""Tests for internal hyperparameter search-space helpers."""

import warnings

import optuna
import pytest

from drevalpy.models import ModelSpec
from drevalpy.models.config import from_spec
from drevalpy.models.config.model import ModelConfig
from drevalpy.models.tuning.search_space import (
    apply_merged_to_model_config,
    defaults_from_merged_space,
    draw_from_optuna_trial,
    extract_defaults,
    merge_model_config_spaces,
    merge_search_spaces,
    split_hyperparameters,
)


def test_merge_split_and_extract_defaults() -> None:
    merged = merge_search_spaces(
        sample_featurizer_space={"n_components": {"type": "int", "low": 4, "high": 64, "default": 8}},
        drug_featurizer_space={"n_bits": {"type": "int", "low": 64, "high": 256, "default": 128}},
        predictor_space={"alpha": {"type": "float", "low": 0.1, "high": 1.0, "default": 0.5}},
    )
    assert set(merged) == {"sample_featurizer.n_components", "drug_featurizer.n_bits", "predictor.alpha"}

    values = {"sample_featurizer.n_components": 8, "drug_featurizer.n_bits": 128, "predictor.alpha": 0.5}
    assert split_hyperparameters(values) == ({"n_components": 8}, {"n_bits": 128}, {"alpha": 0.5})
    assert split_hyperparameters({"alpha": 1.0, "l1_ratio": 0.5})[2] == {"alpha": 1.0, "l1_ratio": 0.5}

    assert extract_defaults(
        sample_featurizer_space={"n_components": {"type": "int", "default": 16}},
        predictor_space={"alpha": {"type": "float", "default": 0.1}},
    ) == {"sample_featurizer.n_components": 16, "predictor.alpha": 0.1}


def test_merge_concat_child_spaces_use_qualified_selectors() -> None:
    config = from_spec("pca[expression]+landmarkGenes:fingerprints:randomForest")
    assert isinstance(config, ModelConfig)
    merged = merge_model_config_spaces(config)
    assert any(key.startswith("sample_featurizer.pca[expression].") for key in merged)
    assert any(key.startswith("sample_featurizer.landmarkGenes.") for key in merged)
    assert any("predictor.randomForest." in key for key in merged)
    # The class-level space is what HPO tunes; it must not drift from the config merge.
    assert ModelSpec("ComboRF", config).hyperparameter_space == merged


def test_merge_same_name_different_views_get_distinct_keys() -> None:
    config = from_spec("pca[expression]+pca[proteomics]:fingerprints:randomForest")
    assert isinstance(config, ModelConfig)
    merged = merge_model_config_spaces(config)
    assert any(key.startswith("sample_featurizer.pca[expression].") for key in merged)
    assert any(key.startswith("sample_featurizer.pca[proteomics].") for key in merged)


def test_apply_merged_to_model_config() -> None:
    config = from_spec("pca[expression]:identity:randomForest")
    assert isinstance(config, ModelConfig)
    merged = defaults_from_merged_space(merge_model_config_spaces(config))

    updated = apply_merged_to_model_config(config, merged)

    assert updated.featurizer_values("sample", "pca[expression]")["n_components"] == 128
    assert all("." in key for key in updated.values)
    with pytest.raises(ValueError, match="Indexed featurizer hyperparameter keys are no longer supported"):
        apply_merged_to_model_config(config, {"sample_featurizer.pca.0.n_components": 8})


class TestDrawFromOptunaTrial:
    @pytest.mark.parametrize(
        ("spec", "is_in_range"),
        [
            pytest.param({"type": "int", "low": 10, "high": 20, "default": 15}, lambda v: 10 <= v <= 20, id="int"),
            pytest.param(
                {"type": "float", "low": 0.1, "high": 0.9, "default": 0.2},
                lambda v: 0.1 <= v <= 0.9,
                id="float",
            ),
            pytest.param(
                {"type": "float", "low": 0.001, "high": 10.0, "log": True, "default": 1.0},
                lambda v: 0.001 <= v <= 10.0,
                id="log-float",
            ),
            pytest.param(
                {"type": "categorical", "choices": ["linear", "rbf", "poly"], "default": "rbf"},
                lambda v: v in {"linear", "rbf", "poly"},
                id="categorical",
            ),
            pytest.param(42, lambda v: v == 42, id="non-mapping-is-passed-through"),
        ],
    )
    def test_each_spec_kind_draws_inside_its_own_domain(self, spec, is_in_range) -> None:
        trial = optuna.create_study().ask()

        drawn = draw_from_optuna_trial(trial, {"param": spec})

        assert is_in_range(drawn["param"])

    def test_list_valued_choices_are_drawn_whole_and_without_an_optuna_warning(self) -> None:
        choices = [[32, 16, 8, 4], [128, 64, 32], [16, 8, 4]]
        spec = {"type": "categorical", "choices": choices, "default": choices[0]}
        trial = optuna.create_study().ask()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            drawn = draw_from_optuna_trial(trial, {"units_per_layer": spec})

        assert drawn["units_per_layer"] in choices

    def test_the_qualified_key_survives_drawing(self) -> None:
        space = {
            "predictor.randomForest.n_estimators": {"type": "int", "low": 10, "high": 20, "default": 15},
            "predictor.randomForest.max_samples": {"type": "float", "low": 0.1, "high": 0.9, "default": 0.2},
        }
        trial = optuna.create_study().ask()

        drawn = draw_from_optuna_trial(trial, space)

        assert set(drawn) == set(space)
