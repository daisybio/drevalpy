"""Drawing hyperparameter configurations from a search space."""

from __future__ import annotations

import pytest

from drevalpy.models.tuning.search_space import sample_configurations

_CATEGORICAL = {"scaler": {"type": "categorical", "choices": ["standard", "minmax"]}}
_MIXED = {
    "n_components": {"type": "int", "low": 2, "high": 4},
    "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    "hidden_dim": {"type": "pow2", "low": 3, "high": 5},
}


class TestEmptySpace:
    def test_an_empty_space_yields_empty_configs(self):
        assert sample_configurations({}, 2) == [{}, {}]

    def test_requesting_zero_configs_from_an_empty_space_yields_nothing(self):
        assert sample_configurations({}, 0) == []


class TestSampling:
    def test_the_requested_number_of_configs_is_returned(self):
        configs = sample_configurations(_CATEGORICAL, 2)

        assert len(configs) == 2

    def test_every_config_covers_the_whole_space(self):
        (config,) = sample_configurations(_MIXED, 1)

        assert set(config) == {"n_components", "learning_rate", "hidden_dim"}

    def test_categorical_values_come_from_the_declared_choices(self):
        configs = sample_configurations(_CATEGORICAL, 2)

        assert all(config["scaler"] in {"standard", "minmax"} for config in configs)

    @pytest.mark.parametrize(
        ("key", "low", "high"),
        [
            pytest.param("n_components", 2, 4, id="int-range"),
            pytest.param("learning_rate", 1e-4, 1e-2, id="log-float-range"),
            pytest.param("hidden_dim", 8, 32, id="pow2-range"),
        ],
    )
    def test_drawn_values_respect_the_declared_bounds(self, key, low, high):
        (config,) = sample_configurations(_MIXED, 1)

        assert low <= config[key] <= high

    def test_integer_parameters_are_drawn_as_integers(self):
        (config,) = sample_configurations(_MIXED, 1)

        assert isinstance(config["n_components"], int)

    def test_pow2_parameters_are_exponentiated(self):
        (config,) = sample_configurations(_MIXED, 1)

        assert config["hidden_dim"] in {8, 16, 32}

    def test_an_unknown_parameter_type_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown hyperparameter type"):
            sample_configurations({"weird": {"type": "quaternion"}}, 1)

    def test_the_same_seed_draws_the_same_configs(self):
        assert sample_configurations(_MIXED, 3, seed=1) == sample_configurations(_MIXED, 3, seed=1)

    def test_empty_configs_are_distinct_objects(self):
        first, second = sample_configurations({}, 2)

        assert first is not second
