"""Small helpers for restoring component state from serialized dicts."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any


def state_float(state: Mapping[str, object], key: str) -> float | None:
    """Return *key* from *state* as float when present.

    :param state: state.
    :param key: key.
    :returns: Result.
    """
    value = state.get(key)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return None


def state_str_dict(state: Mapping[str, object], key: str) -> dict[str, float]:
    """Return a string-keyed float mapping stored under *key*.

    :param state: state.
    :param key: key.
    :returns: Result.
    """
    value = state.get(key)
    if not isinstance(value, dict):
        return {}
    return {str(item_key): float(item_value) for item_key, item_value in value.items()}


def state_str_list(state: Mapping[str, object], key: str) -> list[str] | None:
    """Return a string list stored under *key*.

    :param state: state.
    :param key: key.
    :returns: Result.
    """
    value = state.get(key)
    if not isinstance(value, list):
        return None
    return [str(item) for item in value]


def state_mapping(state: Mapping[str, object], key: str) -> dict[str, Any]:
    """Return a mapping stored under *key*.

    :param state: state.
    :param key: key.
    :returns: Result.
    """
    value = state.get(key)
    if not isinstance(value, dict):
        return {}
    return dict(value)


def state_int(state: Mapping[str, object], key: str) -> int | None:
    """Return *key* from *state* as int when present.

    :param state: state.
    :param key: key.
    :returns: Result.
    """
    value = state.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None
