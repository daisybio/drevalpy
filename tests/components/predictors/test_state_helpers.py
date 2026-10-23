"""Tests for the serialized-state helpers.

Mirrors the private module ``drevalpy.components.predictors._state_helpers``
with the leading underscore stripped, as private-module tests do.
"""

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


def _state(payload: dict[str, object], key: str = "payload") -> dict[str, object]:
    return {key: save_trusted_mapping(payload)}


def test_load_payload_state_returns_the_decoded_payload() -> None:
    payload = {"predictor_hyperparameters": {"epochs": 3}, "model_state": b"weights"}

    assert load_payload_state(_state(payload), "Toy") == payload


def test_load_payload_state_reads_a_custom_key() -> None:
    payload = {"predictor_hyperparameters": {}}

    assert load_payload_state(_state(payload, "checkpoint"), "Toy", key="checkpoint") == payload


def test_load_payload_state_rejects_a_missing_entry() -> None:
    with pytest.raises(PredictorStateError, match="Toy state requires a payload byte blob"):
        load_payload_state({}, "Toy")


def test_load_payload_state_rejects_a_blob_that_is_not_bytes() -> None:
    with pytest.raises(PredictorStateError, match="Toy state requires a checkpoint byte blob"):
        load_payload_state({"checkpoint": "not bytes"}, "Toy", key="checkpoint")


def test_load_payload_state_rejects_an_undecodable_blob() -> None:
    with pytest.raises(PredictorStateError, match="Toy payload could not be deserialized") as caught:
        load_payload_state({"payload": b"garbage"}, "Toy")

    assert caught.value.__cause__ is not None


def test_load_payload_state_requires_predictor_hyperparameters() -> None:
    with pytest.raises(PredictorStateError, match="Toy payload is missing predictor_hyperparameters"):
        load_payload_state(_state({"model_state": b"weights"}), "Toy")


def test_load_payload_state_rejects_hyperparameters_that_are_not_a_mapping() -> None:
    with pytest.raises(PredictorStateError, match="missing predictor_hyperparameters"):
        load_payload_state(_state({"predictor_hyperparameters": [1, 2]}), "Toy")
