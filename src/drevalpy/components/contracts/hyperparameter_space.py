"""Validate structured hyperparameter search-space specs, read their defaults, and the tunable-component hooks.

:class:`TunableComponentMixin` lives beside the validator it calls rather than in
either component package: ``Featurizer`` and ``Predictor`` are siblings under
``components/``, so a home inside one of them would make the other depend on it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "TunableComponentMixin",
    "default_values",
    "validate_component_hyperparameter_space",
    "validate_hyperparameter_space",
]


def _classify_specs(space: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Split *space* into the entries that are not mappings and those without a default.

    :param space: Hyperparameter space to inspect.
    :returns: ``(invalid, missing)`` name lists.
    """
    invalid: list[str] = []
    missing: list[str] = []
    for name, spec in space.items():
        if not isinstance(spec, Mapping):
            invalid.append(str(name))
        elif "default" not in spec:
            missing.append(str(name))
    return invalid, missing


def _describe_problems(invalid: list[str], missing: list[str]) -> str:
    """Phrase the offending names, sorted, for the error message.

    :param invalid: Names whose spec is not a mapping.
    :param missing: Names whose spec declares no ``default``.
    :returns: The problem clauses joined for interpolation into the message.
    """
    parts: list[str] = []
    if invalid:
        parts.append("non-mapping specs for " + ", ".join(repr(name) for name in sorted(invalid)))
    if missing:
        parts.append("missing 'default' for " + ", ".join(repr(name) for name in sorted(missing)))
    return "; ".join(parts)


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
    invalid, missing = _classify_specs(space)
    if invalid or missing:
        msg = f"Invalid hyperparameter space in {context}: " + _describe_problems(invalid, missing)
        raise ValueError(msg)


def default_values(space: Mapping[str, Any] | None, *, context: str) -> dict[str, Any]:
    """Validate *space* and return the ``default`` of every entry.

    :param space: Local or merged hyperparameter space, or ``None`` / empty.
    :param context: Label included in error messages (class or config field).
    :returns: Parameter names mapped to their declared ``default`` values.
    """
    validate_hyperparameter_space(space, context=context)
    return {name: spec["default"] for name, spec in (space or {}).items()}


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


class TunableComponentMixin:
    """The HPO hooks every component kind declares identically.

    Both ``Featurizer`` and ``Predictor`` are tuned by the same search-space
    grammar. The default here is the no-op end of the contract: a component with
    nothing to tune returns an empty space.

    Subclasses override ``get_hyperparameter_space`` to declare what is tunable.
    ``get_default_hyperparameters`` is not an override point - it reads whatever the
    space declares.
    """

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs for HPO.

        :returns: Mapping of parameter name to search-space spec dicts.
        """
        return {}

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameter values from the HP space.

        :returns: Parameter names mapped to their declared ``default`` values.
        """
        return default_values(cls.get_hyperparameter_space(), context=f"{cls.__name__}.get_hyperparameter_space()")
