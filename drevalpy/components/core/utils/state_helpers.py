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
