"""Small helpers for restoring component state from serialized dicts."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.utils.torch_io import load_trusted_mapping


def state_float(state: Mapping[str, object], key: str) -> float | None:
    """Return *key* from *state* as float when present.

    :param state: Serialized component state.
    :param key: Entry to read.
    :returns: The entry as a float (numeric strings are parsed; an unparsable string raises ``ValueError``),
        or ``None`` if it is absent or not numeric.
    """
    value = state.get(key)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return None


def state_mapping(state: Mapping[str, object], key: str) -> dict[str, Any]:
    """Return a mapping stored under *key*.

    :param state: Serialized component state.
    :param key: Entry to read.
    :returns: A copy of the stored mapping, or an empty dict if the entry is absent or not a mapping.
    """
    value = state.get(key)
    if not isinstance(value, dict):
        return {}
    return dict(value)


def load_payload_state(state: Mapping[str, Any], owner: str, key: str = "payload") -> dict[str, Any]:
    """Decode the serialized payload a predictor's ``get_state`` stored under *key*.

    :param state: State mapping handed to ``set_state``.
    :param owner: Predictor name used in error messages.
    :param key: Entry of *state* that holds the payload byte blob.
    :returns: The deserialized payload, which is guaranteed to hold a ``predictor_hyperparameters`` dict.
    :raises PredictorStateError: If the entry is missing or not bytes, cannot be deserialized, or the payload
        has no ``predictor_hyperparameters`` dict.
    """
    blob = state.get(key)
    if not isinstance(blob, (bytes, bytearray)):
        msg = f"{owner} state requires a {key} byte blob"
        raise PredictorStateError(msg)
    try:
        payload = load_trusted_mapping(bytes(blob))
    except Exception as exc:
        msg = f"{owner} {key} could not be deserialized"
        raise PredictorStateError(msg) from exc
    if not isinstance(payload.get("predictor_hyperparameters"), dict):
        msg = f"{owner} {key} is missing predictor_hyperparameters"
        raise PredictorStateError(msg)
    return payload
