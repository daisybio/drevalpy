"""Tests for the serialized-state helpers."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors._state_helpers import load_payload_state, state_float, state_mapping
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.utils.torch_io import save_trusted_mapping


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


def test_state_float_is_none_for_missing_or_non_numeric_values_and_raises_on_unparsable_strings() -> None:
    assert state_float({}, "dataset_mean") is None
    assert state_float({"dataset_mean": [1.0]}, "dataset_mean") is None
    with pytest.raises(ValueError):
        state_float({"dataset_mean": "not-a-number"}, "dataset_mean")


def test_state_mapping_returns_a_copy_and_empty_for_missing_or_non_mapping_values() -> None:
    stored = {"alpha": 1.0, "mode": "regression"}

    result = state_mapping({"hyperparameters": stored}, "hyperparameters")

    assert result == stored
    assert result is not stored
    assert state_mapping({}, "hyperparameters") == {}
    assert state_mapping({"hyperparameters": ["alpha"]}, "hyperparameters") == {}


def _state(payload: dict[str, object], key: str = "payload") -> dict[str, object]:
    return {key: save_trusted_mapping(payload)}


def test_load_payload_state_returns_the_decoded_payload_under_any_key() -> None:
    payload = {"predictor_hyperparameters": {"epochs": 3}, "model_state": b"weights"}

    assert load_payload_state(_state(payload), "Toy") == payload
    assert load_payload_state(_state(payload, "checkpoint"), "Toy", key="checkpoint") == payload


@pytest.mark.parametrize(
    ("state", "key", "message"),
    [
        ({}, "payload", "Toy state requires a payload byte blob"),
        ({"checkpoint": "not bytes"}, "checkpoint", "Toy state requires a checkpoint byte blob"),
        ({"payload": b"garbage"}, "payload", "Toy payload could not be deserialized"),
        (_state({"model_state": b"weights"}), "payload", "Toy payload is missing predictor_hyperparameters"),
        (_state({"predictor_hyperparameters": [1, 2]}), "payload", "missing predictor_hyperparameters"),
    ],
)
def test_load_payload_state_rejects_malformed_state(state: dict[str, object], key: str, message: str) -> None:
    with pytest.raises(PredictorStateError, match=message):
        load_payload_state(state, "Toy", key=key)
