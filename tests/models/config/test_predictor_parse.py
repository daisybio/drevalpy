"""Tests for compact predictor config parsing."""

from __future__ import annotations

import pytest

from drevalpy.components.registry.register_builtins import register_builtin_components
from drevalpy.models.config._predictor_parse import normalize_predictor_config


@pytest.fixture(autouse=True)
def _registry() -> None:
    """Register the built-in components the one-key notation looks up."""
    register_builtin_components()


def test_normalize_string_shorthand() -> None:
    payload = normalize_predictor_config("randomForest")
    assert payload == {"name": "randomForest"}


def test_normalize_one_key_mapping() -> None:
    payload = normalize_predictor_config({"randomForest": {"n_estimators": 10}})
    assert payload["name"] == "randomForest"
    assert payload["hyperparameter_space"]["n_estimators"]["default"] == 10


def test_normalize_one_key_mapping_may_be_null() -> None:
    assert normalize_predictor_config({"randomForest": None}) == {"name": "randomForest"}


def test_normalize_canonical_mapping_passes_through() -> None:
    """A mapping that already names its predictor normalizes to itself."""
    space = {"n_estimators": {"type": "int", "low": 2, "high": 20, "default": 4}}
    payload = normalize_predictor_config({"name": "randomForest", "hyperparameter_space": space})
    assert payload == {"name": "randomForest", "hyperparameter_space": space}


def test_explicit_space_wins_over_a_loose_value() -> None:
    space = {"n_estimators": {"type": "int", "low": 2, "high": 20, "default": 4}}
    payload = normalize_predictor_config(
        {"randomForest": {"hyperparameter_space": space, "n_estimators": 10}},
    )
    assert payload["hyperparameter_space"]["n_estimators"]["default"] == 4


def test_non_tunable_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="do not accept non-tunable options \\('not_a_knob'\\)"):
        normalize_predictor_config({"randomForest": {"not_a_knob": 1}})


def test_one_key_body_must_be_a_mapping() -> None:
    with pytest.raises(ValueError, match="must be a mapping when provided"):
        normalize_predictor_config({"randomForest": 5})


def test_normalize_rejects_invalid_shape() -> None:
    with pytest.raises(TypeError, match="string or mapping"):
        normalize_predictor_config(123)


def test_mapping_without_name_or_one_key_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="string, one-key mapping, or dict with 'name'"):
        normalize_predictor_config({"hyperparameter_space": {}, "extra": 1})
