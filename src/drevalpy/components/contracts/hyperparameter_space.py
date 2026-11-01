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
    "admits",
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


def _plain(value: Any) -> Any:
    """Return *value* with tuples turned into lists, recursively, so frozen and YAML values compare equal."""
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _is_number(value: Any, *, integer: bool) -> bool:
    """Return whether *value* is a real number (an integer if *integer*); ``bool`` is neither."""
    if isinstance(value, bool):
        return False
    return isinstance(value, int) if integer else isinstance(value, (int, float))


def admits(spec: Any, value: Any) -> bool:
    """Return whether *value* lies in the search-space entry *spec*, i.e. whether tuning could draw it.

    ``fixed`` entries and non-mapping entries admit any value: they are never drawn. ``int`` and
    ``float`` entries admit numbers in ``[low, high]``, ``pow2`` entries the powers ``2**low`` to
    ``2**high``, and ``categorical`` entries an element of ``choices`` (tuples and lists compare equal).
    Only the bounds and choices an entry declares are checked.

    :param spec: One search-space entry.
    :param value: A hyperparameter value.
    :returns: ``True`` when *value* is inside *spec*.
    """
    if not isinstance(spec, Mapping) or is_fixed(spec):
        return True
    kind = spec.get("type", "categorical")
    if kind == "categorical":
        return "choices" not in spec or _plain(value) in [_plain(choice) for choice in spec["choices"]]
    if kind not in ("int", "float", "pow2"):
        return True
    if not _is_number(value, integer=kind != "float"):
        return False
    if kind == "pow2":
        if value <= 0 or value & (value - 1):
            return False
        value = value.bit_length() - 1
    return spec.get("low", value) <= value <= spec.get("high", value)


def _classify_specs(space: Mapping[str, Any]) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split *space* into invalid, default-less, malformed fixed and out-of-range-default entries.

    An entry is invalid if it is not a mapping, and out of range if its ``default`` lies outside it.

    :param space: Hyperparameter space to inspect.
    :returns: ``(invalid, missing, bad_fixed, outside)`` name lists.
    """
    invalid: list[str] = []
    missing: list[str] = []
    bad_fixed: list[str] = []
    outside: list[str] = []
    for name, spec in space.items():
        if not isinstance(spec, Mapping):
            invalid.append(str(name))
            continue
        if "default" not in spec:
            missing.append(str(name))
        elif not admits(spec, spec["default"]):
            outside.append(str(name))
        if is_fixed(spec) and any(key in spec for key in _SEARCH_KEYS):
            bad_fixed.append(str(name))
    return invalid, missing, bad_fixed, outside


def _describe_problems(invalid: list[str], missing: list[str], bad_fixed: list[str], outside: list[str]) -> str:
    """Phrase the offending names, sorted, for the error message.

    :param invalid: Names whose spec is not a mapping.
    :param missing: Names whose spec declares no ``default``.
    :param bad_fixed: Names of fixed specs that also declare search keys.
    :param outside: Names whose ``default`` lies outside their own range or choices.
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
    if outside:
        parts.append("'default' outside the range or choices for " + ", ".join(repr(name) for name in sorted(outside)))
    return "; ".join(parts)


def validate_hyperparameter_space(
    space: Mapping[str, Any] | None,
    *,
    context: str,
) -> None:
    """Reject search-space entries that are not mappings with a ``default`` inside their own range.

    Every parameter must declare a concrete ``default`` so model construction
    without overrides can resolve a complete value mapping, and tuning must be able
    to draw it (see :func:`admits`). A ``{"type": "fixed",
    "default": v}`` entry declares a value HPO never draws; it may not carry
    ``low``/``high``/``choices``/``log``.

    :param space: Local or merged hyperparameter space, or ``None`` / empty.
    :param context: Label included in error messages (class or config field).
    :raises ValueError: If any entry is not a mapping, lacks ``default``, has a ``default`` outside its range or
        choices, or is a malformed fixed entry.
    """
    if not space:
        return
    problems = _classify_specs(space)
    if any(problems):
        msg = f"Invalid hyperparameter space in {context}: " + _describe_problems(*problems)
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
