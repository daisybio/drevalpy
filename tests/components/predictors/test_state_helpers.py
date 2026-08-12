"""Tests for the serialized-state coercion helpers.

Mirrors the private module ``drevalpy.components.predictors._state_helpers``
with the leading underscore stripped, per the all-private mirroring rule in
``AGENTS.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors._state_helpers import state_float, state_mapping


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        pytest.param(2, 2.0, id="int"),
        pytest.param(2.5, 2.5, id="float"),
        pytest.param(np.float32(1.5), 1.5, id="numpy-scalar"),
        pytest.param("3.25", 3.25, id="numeric-string"),
        pytest.param(True, 1.0, id="bool-is-real"),
    ],
)
def test_state_float_coerces_real_and_string_values(stored: object, expected: float) -> None:
    result = state_float({"key": stored}, "key")

    assert result == pytest.approx(expected)
    assert isinstance(result, float)


def test_state_float_returns_none_for_missing_key() -> None:
    assert state_float({}, "dataset_mean") is None


def test_state_float_returns_none_for_non_numeric_value() -> None:
    assert state_float({"dataset_mean": [1.0]}, "dataset_mean") is None


def test_state_float_raises_on_unparsable_string() -> None:
    with pytest.raises(ValueError):
        state_float({"dataset_mean": "not-a-number"}, "dataset_mean")


def test_state_mapping_returns_a_copy_of_the_stored_dict() -> None:
    stored = {"alpha": 1.0}

    result = state_mapping({"hyperparameters": stored}, "hyperparameters")

    assert result == {"alpha": 1.0}
    assert result is not stored


def test_state_mapping_stringifies_nothing_but_preserves_keys() -> None:
    result = state_mapping({"hyperparameters": {"n_estimators": 5, "mode": "regression"}}, "hyperparameters")

    assert result == {"n_estimators": 5, "mode": "regression"}


def test_state_mapping_returns_empty_dict_for_missing_key() -> None:
    assert state_mapping({}, "hyperparameters") == {}


def test_state_mapping_returns_empty_dict_for_non_mapping_value() -> None:
    assert state_mapping({"hyperparameters": ["alpha"]}, "hyperparameters") == {}
