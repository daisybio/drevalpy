"""The search space of a model and drawing values from it."""

from __future__ import annotations

import warnings

import optuna
import pytest
from optuna.trial import FixedTrial

from drevalpy.models import ModelSpec
from drevalpy.models.config import from_recipe
from drevalpy.models.config.model import ModelConfig
from drevalpy.models.tuning.search_space import draw, space


def test_space_qualifies_every_leaf_of_a_concat_and_the_predictor() -> None:
    config = from_recipe("pca[expression]+landmarkGenes:fingerprints:randomForest")

    merged = space(config)

    assert any(key.startswith("sample_featurizer.pca[expression].") for key in merged)
    assert any(key.startswith("sample_featurizer.landmarkGenes.") for key in merged)
    assert any(key.startswith("drug_featurizer.fingerprints.") for key in merged)
    assert any(key.startswith("predictor.randomForest.") for key in merged)
    assert not any(key.startswith("sample_featurizer.concatFeaturizers.") for key in merged)
    assert ModelSpec("ComboRF", config).hyperparameter_space == merged


def test_space_orders_sample_then_drug_then_predictor() -> None:
    slots = [key.split(".", 1)[0] for key in space(from_recipe("pca[expression]:fingerprints:elasticNet"))]

    assert slots == sorted(slots, key=["sample_featurizer", "drug_featurizer", "predictor"].index)


def test_a_declared_space_replaces_the_class_space() -> None:
    config = ModelConfig.model_validate(
        {
            "sample_featurizer": {"pca[expression]": {"n_components": 7}},
            "drug_featurizer": "fingerprints",
            "predictor": "elasticNet",
        }
    )

    assert space(config)["sample_featurizer.pca[expression].n_components"]["default"] == 7
    assert ModelSpec("custom", config).hyperparameters["sample_featurizer.pca[expression].n_components"] == 7


def test_feature_free_models_have_an_empty_space() -> None:
    assert space(from_recipe("naiveMean")) == {}


class TestDraw:
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
            pytest.param({"type": "pow2", "low": 3, "high": 5, "default": 16}, lambda v: v in {8, 16, 32}, id="pow2"),
            pytest.param(42, lambda v: v == 42, id="non-mapping-is-passed-through"),
        ],
    )
    def test_each_spec_kind_draws_inside_its_own_domain(self, spec, is_in_range) -> None:
        trial = optuna.create_study().ask()

        drawn = draw(trial, {"param": spec})

        assert is_in_range(drawn["param"])

    def test_a_fixed_entry_keeps_its_default_without_asking_the_trial(self) -> None:
        trial = optuna.create_study().ask()

        drawn = draw(trial, {"epochs": {"type": "fixed", "default": 3}})

        assert drawn == {"epochs": 3}
        assert trial.params == {}

    def test_list_valued_choices_are_drawn_whole_and_without_an_optuna_warning(self) -> None:
        choices = [[32, 16, 8, 4], [128, 64, 32], [16, 8, 4]]
        spec = {"type": "categorical", "choices": choices, "default": choices[0]}
        trial = optuna.create_study().ask()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            drawn = draw(trial, {"units_per_layer": spec})

        assert drawn["units_per_layer"] in choices

    def test_replaying_the_sampled_params_gives_the_drawn_values(self) -> None:
        merged = {
            "predictor.randomForest.n_estimators": {"type": "int", "low": 10, "high": 20, "default": 15},
            "drug_featurizer.fingerprints.n_bits": {"type": "pow2", "low": 9, "high": 11, "default": 2048},
        }
        trial = optuna.create_study().ask()

        drawn = draw(trial, merged)

        assert set(drawn) == set(merged)
        assert draw(FixedTrial(trial.params), merged) == drawn

    def test_an_unknown_type_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown hyperparameter type 'nope'"):
            draw(optuna.create_study().ask(), {"param": {"type": "nope"}})
