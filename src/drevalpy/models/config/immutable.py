"""Deep-freeze helpers for immutable model configuration values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import AfterValidator, PlainSerializer


def freeze_value(value: Any) -> Any:
    """Recursively freeze mappings and sequences into immutable containers.

    :param value: Arbitrary nested value from config construction.
    :returns: Immutable view of *value* (scalars unchanged).
    """
    if isinstance(value, MappingProxyType):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_value(item) for key, item in value.items()})
    if isinstance(value, (str, bytes, bytearray)):
        return value
    if isinstance(value, Sequence):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(freeze_value(item) for item in value)
    return value


def thaw_value(value: Any) -> Any:
    """Recursively convert frozen containers back into plain dicts and lists.

    Used by ``model_dump`` serializers so persistence and YAML stay JSON-friendly.

    :param value: Possibly frozen nested value.
    :returns: Mutable plain-Python equivalent.
    """
    if isinstance(value, Mapping):
        return {key: thaw_value(item) for key, item in value.items()}
    if isinstance(value, (str, bytes, bytearray)):
        return value
    if isinstance(value, tuple):
        return [thaw_value(item) for item in value]
    if isinstance(value, frozenset):
        return [thaw_value(item) for item in value]
    return value


FrozenMapping = Annotated[
    Mapping[str, Any],
    AfterValidator(freeze_value),
    PlainSerializer(thaw_value, return_type=dict),
]
"""Untyped config mapping that is frozen on validation and thawed on dump.

Pydantic's ``frozen=True`` is shallow, so the arbitrary nested contents of these
escape-hatch fields need the recursive walk in :func:`freeze_value`; ``model_dump``
must still hand out plain dicts and lists for YAML and checkpoint persistence.
"""
