"""Tests for internal hyperparameter search-space helpers."""

import optuna
import pytest

from drevalpy.models import construct_model
from drevalpy.models.config import from_spec
from drevalpy.models.config.model import ModelConfig
from drevalpy.models.tuning.search_space import (
    apply_merged_to_model_config,
    defaults_from_merged_space,
    extract_defaults,
    merge_model_config_spaces,
    merge_search_spaces,
    sample_from_optuna_trial,
    split_hyperparameters,
)
from drevalpy.registry._builtins import register_builtin_components


def test_merge_all_three_spaces() -> None:
    merged = merge_search_spaces(
        cell_line_featurizer_space={"n_components": {"type": "int", "low": 4, "high": 64, "default": 8}},
        drug_featurizer_space={"n_bits": {"type": "int", "low": 64, "high": 256, "default": 128}},
        predictor_space={"alpha": {"type": "float", "low": 0.1, "high": 1.0, "default": 0.5}},
    )
    assert set(merged) == {
        "cell_line_featurizer.n_components",
        "drug_featurizer.n_bits",
        "predictor.alpha",
    }


def test_split_hyperparameters_inverts_merge() -> None:
    merged = {
        "cell_line_featurizer.n_components": 8,
        "drug_featurizer.n_bits": 128,
        "predictor.alpha": 0.5,
    }
    cell_line_hp, drug_hp, predictor_hp = split_hyperparameters(merged)
    assert cell_line_hp == {"n_components": 8}
    assert drug_hp == {"n_bits": 128}
    assert predictor_hp == {"alpha": 0.5}


def test_split_predictor_only_fallback() -> None:
    merged = {"alpha": 1.0, "l1_ratio": 0.5}
    _, _, predictor_hp = split_hyperparameters(merged)
    assert predictor_hp == {"alpha": 1.0, "l1_ratio": 0.5}


def test_merge_concat_child_spaces_use_qualified_selectors() -> None:
    register_builtin_components()
    spec = "pca[expression]+landmarkGenes:fingerprints:randomForest"
    config = from_spec(spec)
    assert isinstance(config, ModelConfig)
    merged = merge_model_config_spaces(config)
    pca_keys = [key for key in merged if key.startswith("cell_line_featurizer.pca[expression].")]
    assert pca_keys
    assert any(key.startswith("cell_line_featurizer.landmarkGenes.") for key in merged)
    assert any("predictor.randomForest." in key for key in merged)
    # The class-level space is what HPO tunes; it must not drift from the config merge.
    assert construct_model("ComboRF", spec).get_structured_hyperparameter_space() == merged


def test_merge_same_name_different_views_get_distinct_keys() -> None:
    register_builtin_components()
    config = from_spec("pca[expression]+pca[proteomics]:fingerprints:randomForest")
    assert isinstance(config, ModelConfig)
    merged = merge_model_config_spaces(config)
    assert any(key.startswith("cell_line_featurizer.pca[expression].") for key in merged)
    assert any(key.startswith("cell_line_featurizer.pca[proteomics].") for key in merged)


def test_apply_rejects_indexed_featurizer_keys() -> None:
    register_builtin_components()
    config = from_spec("pca[expression]:identity:randomForest")
    assert isinstance(config, ModelConfig)
    with pytest.raises(
        ValueError,
        match="Indexed featurizer hyperparameter keys are no longer supported",
    ):
        apply_merged_to_model_config(
            config,
            {"cell_line_featurizer.pca.0.n_components": 8},
        )


def test_apply_merged_to_model_config_strips_featurizer_prefix() -> None:
    register_builtin_components()
    config = from_spec("pca[expression]:identity:randomForest")
    assert isinstance(config, ModelConfig)
    merged = defaults_from_merged_space(merge_model_config_spaces(config))
    updated = apply_merged_to_model_config(config, merged)
    assert updated.featurizer_values("cell_line", "pca[expression]")["n_components"] == 128
    assert all("." in key for key in updated.values)


def test_extract_defaults() -> None:
    defaults = extract_defaults(
        cell_line_featurizer_space={"n_components": {"type": "int", "default": 16}},
        predictor_space={"alpha": {"type": "float", "default": 0.1}},
    )
    assert defaults == {
        "cell_line_featurizer.n_components": 16,
        "predictor.alpha": 0.1,
    }


class TestSampleFromOptunaTrial:
    """One spec kind per case; the four suggest_* branches shared one skeleton."""

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
    def test_each_spec_kind_samples_inside_its_own_domain(self, spec, is_in_range) -> None:
        trial = optuna.create_study().ask()

        sampled = sample_from_optuna_trial(trial, {"param": spec})

        assert is_in_range(sampled["param"])

    def test_the_qualified_key_survives_sampling(self) -> None:
        """Dotted selectors are what ``split_hyperparameters`` later routes on."""
        space = {
            "predictor.randomForest.n_estimators": {"type": "int", "low": 10, "high": 20, "default": 15},
            "predictor.randomForest.max_samples": {"type": "float", "low": 0.1, "high": 0.9, "default": 0.2},
        }
        trial = optuna.create_study().ask()

        sampled = sample_from_optuna_trial(trial, space)

        assert set(sampled) == set(space)
