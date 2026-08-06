"""Deep-freeze helpers for immutable model configuration values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Protocol, TypeVar, cast


class _PydanticModel(Protocol):
    def model_dump(self, *, mode: str = "python") -> dict[str, Any]: ...

    @classmethod
    def model_validate(cls, obj: Any) -> Any: ...


_T = TypeVar("_T", bound=_PydanticModel)


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


def rebuild_model(model: _T, **updates: Any) -> _T:
    """Return a validated replacement for a Pydantic model with field updates.

    Pydantic ``model_copy(update=...)`` skips validators; this helper dumps,
    merges updates, and re-validates so frozen/semantic invariants always run.

    :param model: Existing Pydantic model instance.
    :param updates: Field overrides to apply.
    :returns: Newly validated model of the same type.
    """
    payload = model.model_dump(mode="python")
    payload.update(updates)
    return cast(_T, type(model).model_validate(payload))
