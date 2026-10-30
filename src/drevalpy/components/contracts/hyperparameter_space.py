"""Validate structured hyperparameter search-space specs, read their defaults, and the tunable-component hooks.

:class:`TunableComponent` lives beside the validator it calls rather than in
either component package: ``Featurizer`` and ``Predictor`` are siblings under
``components/``, so a home inside one of them would make the other depend on it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "FIXED",
    "TunableComponent",
    "default_values",
    "fixed_entries",
    "is_fixed",
    "tunable_entries",
    "validate_component_hyperparameter_space",
    "validate_hyperparameter_space",
]


FIXED = "fixed"
_SEARCH_KEYS = ("low", "high", "choices", "log")


def is_fixed(spec: Mapping[str, Any]) -> bool:
    """Return whether *spec* is a ``{"type": "fixed", "default": v}`` entry that HPO never draws.

    :param spec: One search-space entry.
    :returns: ``True`` for a fixed entry.
    """
    return spec.get("type") == FIXED


def fixed_entries(values: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Declare each of *values* as a fixed search-space entry.

    :param values: Parameter names mapped to the value HPO must leave untouched.
    :returns: ``{"type": "fixed", "default": value}`` per name.
    """
    return {name: {"type": FIXED, "default": value} for name, value in values.items()}


def tunable_entries(space: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the entries of *space* that HPO draws, dropping fixed ones.

    :param space: Hyperparameter space, or ``None`` / empty.
    :returns: The non-fixed entries, in declaration order.
    """
    return {name: spec for name, spec in (space or {}).items() if not is_fixed(spec)}


def _classify_specs(space: Mapping[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Split *space* into the entries that are not mappings, lack a default, or are malformed fixed entries.

    :param space: Hyperparameter space to inspect.
    :returns: ``(invalid, missing, bad_fixed)`` name lists.
    """
    invalid: list[str] = []
    missing: list[str] = []
    bad_fixed: list[str] = []
    for name, spec in space.items():
        if not isinstance(spec, Mapping):
            invalid.append(str(name))
            continue
        if "default" not in spec:
            missing.append(str(name))
        if is_fixed(spec) and any(key in spec for key in _SEARCH_KEYS):
            bad_fixed.append(str(name))
    return invalid, missing, bad_fixed


def _describe_problems(invalid: list[str], missing: list[str], bad_fixed: list[str]) -> str:
    """Phrase the offending names, sorted, for the error message.

    :param invalid: Names whose spec is not a mapping.
    :param missing: Names whose spec declares no ``default``.
    :param bad_fixed: Names of fixed specs that also declare search keys.
    :returns: The problem clauses joined for interpolation into the message.
    """
    parts: list[str] = []
    if invalid:
        parts.append("non-mapping specs for " + ", ".join(repr(name) for name in sorted(invalid)))
    if missing:
        parts.append("missing 'default' for " + ", ".join(repr(name) for name in sorted(missing)))
    if bad_fixed:
        parts.append(
            "'fixed' specs must not declare low/high/choices/log for "
            + ", ".join(repr(name) for name in sorted(bad_fixed))
        )
    return "; ".join(parts)


def validate_hyperparameter_space(
    space: Mapping[str, Any] | None,
    *,
    context: str,
) -> None:
    """Reject search-space entries that are not mappings with a ``default``.

    Every parameter must declare a concrete ``default`` so model construction
    without overrides can resolve a complete value mapping. A ``{"type": "fixed",
    "default": v}`` entry declares a value HPO never draws; it may not carry
    ``low``/``high``/``choices``/``log``.

    :param space: Local or merged hyperparameter space, or ``None`` / empty.
    :param context: Label included in error messages (class or config field).
    :raises ValueError: If any entry is not a mapping, lacks ``default`` or is a malformed fixed entry.
    """
    if not space:
        return
    invalid, missing, bad_fixed = _classify_specs(space)
    if invalid or missing or bad_fixed:
        msg = f"Invalid hyperparameter space in {context}: " + _describe_problems(invalid, missing, bad_fixed)
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


class TunableComponent:
    """The HPO hooks every component kind declares identically.

    Both ``Featurizer`` and ``Predictor`` are tuned by the same search-space
    grammar. The default here is the no-op end of the contract: a component with
    nothing to tune returns an empty space.

    Subclasses override ``get_hyperparameter_space`` to declare what is tunable.
    ``default_hyperparameters`` is not an override point - it reads whatever the
    space declares, including its ``fixed`` entries.
    """

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs for HPO.

        :returns: Mapping of parameter name to search-space spec dicts.
        """
        return {}

    @classmethod
    def default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameter values from the HP space.

        :returns: Parameter names mapped to their declared ``default`` values.
        """
        return default_values(cls.get_hyperparameter_space(), context=f"{cls.__name__}.get_hyperparameter_space()")
