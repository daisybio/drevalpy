"""Validate structured hyperparameter search-space specs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def validate_hyperparameter_space(
    space: Mapping[str, Any] | None,
    *,
    context: str,
) -> None:
    """Reject search-space entries that are not mappings with a ``default``.

    Every tunable parameter must declare a concrete ``default`` so model
    construction without overrides can resolve a complete value mapping.

    :param space: Local or merged hyperparameter space, or ``None`` / empty.
    :param context: Label included in error messages (class or config field).
    :raises ValueError: If any entry is not a mapping or lacks ``default``.
    """
    if not space:
        return
    missing: list[str] = []
    invalid: list[str] = []
    for name, spec in space.items():
        if not isinstance(spec, Mapping):
            invalid.append(str(name))
            continue
        if "default" not in spec:
            missing.append(str(name))
    if invalid or missing:
        parts: list[str] = []
        if invalid:
            parts.append(
                "non-mapping specs for " + ", ".join(repr(name) for name in sorted(invalid)),
            )
        if missing:
            parts.append(
                "missing 'default' for " + ", ".join(repr(name) for name in sorted(missing)),
            )
        msg = f"Invalid hyperparameter space in {context}: " + "; ".join(parts)
        raise ValueError(msg)


def validate_component_hyperparameter_space(name: str, cls: type[Any]) -> None:
    """Validate ``cls.get_hyperparameter_space()`` at registration time.

    :param name: Registry name under which *cls* is being registered.
    :param cls: Component class exposing ``get_hyperparameter_space``.
    """
    getter = getattr(cls, "get_hyperparameter_space", None)
    if not callable(getter):
        return
    validate_hyperparameter_space(
        getter(),
        context=f"{name!r} ({cls.__name__}.get_hyperparameter_space())",
    )
