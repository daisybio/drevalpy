"""Tests for Optuna-backed hyperparameter sampling.

Each call creates a real Optuna study, so the tests stay at one or two trials:
the point is that declared distributions are respected and that an empty space
short-circuits before a study is created.
"""

from __future__ import annotations

import pytest

from drevalpy.types.data.dataset_utils.sampling import _sample_hp_configs


def _featurizer(space: dict) -> type:
    """Build a throwaway class exposing *space* as its hyperparameter space."""
    return type("StubFeaturizer", (), {"get_hyperparameter_space": classmethod(lambda cls: space)})


_CATEGORICAL = _featurizer({"scaler": {"type": "categorical", "choices": ["standard", "minmax"]}})
_MIXED = _featurizer(
    {
        "n_components": {"type": "int", "low": 2, "high": 4},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "hidden_dim": {"type": "pow2", "low": 3, "high": 5},
    }
)


class TestEmptySpace:
    def test_a_featurizer_without_hyperparameters_yields_empty_configs(self):
        assert _sample_hp_configs(_featurizer({}), 2) == [{}, {}]

    def test_requesting_zero_configs_from_an_empty_space_yields_nothing(self):
        assert _sample_hp_configs(_featurizer({}), 0) == []


class TestSampling:
    def test_the_requested_number_of_configs_is_returned(self):
        configs = _sample_hp_configs(_CATEGORICAL, 2)

        assert len(configs) == 2

    def test_every_config_covers_the_whole_space(self):
        (config,) = _sample_hp_configs(_MIXED, 1)

        assert set(config) == {"n_components", "learning_rate", "hidden_dim"}

    def test_categorical_values_come_from_the_declared_choices(self):
        configs = _sample_hp_configs(_CATEGORICAL, 2)

        assert all(config["scaler"] in {"standard", "minmax"} for config in configs)

    @pytest.mark.parametrize(
        ("key", "low", "high"),
        [
            pytest.param("n_components", 2, 4, id="int-range"),
            pytest.param("learning_rate", 1e-4, 1e-2, id="log-float-range"),
            pytest.param("hidden_dim", 8, 32, id="pow2-range"),
        ],
    )
    def test_sampled_values_respect_the_declared_bounds(self, key, low, high):
        (config,) = _sample_hp_configs(_MIXED, 1)

        assert low <= config[key] <= high

    def test_integer_parameters_are_sampled_as_integers(self):
        (config,) = _sample_hp_configs(_MIXED, 1)

        assert isinstance(config["n_components"], int)

    def test_pow2_parameters_are_exponentiated(self):
        (config,) = _sample_hp_configs(_MIXED, 1)

        assert config["hidden_dim"] in {8, 16, 32}

    def test_an_unknown_parameter_type_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown hyperparameter type"):
            _sample_hp_configs(_featurizer({"weird": {"type": "quaternion"}}), 1)
