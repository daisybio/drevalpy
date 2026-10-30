"""The featurizer registry: one :class:`Registry` for both sides, its ``register`` decorator and the lookup.

A featurizer is registered once under a unique name. The sides it can be used on are a
value it declares (``Featurizer.sides``), not a registry: :func:`get_featurizer` checks a
requested side against that declaration, so a config slot picks the side and the class says
whether it supports it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from drevalpy.components.contracts.hyperparameter_space import validate_component_hyperparameter_space
from drevalpy.registry._registry import Registry
from drevalpy.registry.components._metadata import LiteratureReference
from drevalpy.registry.components._register import register_component
from drevalpy.registry.components._registration_metadata import normalize_registration_metadata
from drevalpy.registry.featurizer._validate import validate_featurizer_declarations
from drevalpy.types.data.feature_source import Side

C = TypeVar("C", bound=type[Any])

#: Every registered featurizer class, in registration order.
featurizers: Registry[type[Any]] = Registry("featurizer")

_SIDES: frozenset[str] = frozenset({"sample", "drug"})


def register(
    name: str,
    *,
    description: str,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
) -> Callable[[C], C]:
    """Return a class decorator that registers a featurizer under *name*.

    The decorator sets ``name`` on the class, and ``storage_key`` to *name* when neither the
    class nor a base declares one. The sides come from the class's ``sides`` declaration.

    :param name: Unique registry name used in model configs and listings.
    :param description: Short human-readable summary.
    :param tags: Optional discovery tags.
    :param reference: Optional literature citation metadata.
    :returns: Class decorator that registers and returns the decorated class.
    """
    metadata = normalize_registration_metadata(description, tags, reference)

    def validate(registered_name: str, cls: type[Any]) -> None:
        validate_component_hyperparameter_space(registered_name, cls)
        validate_featurizer_declarations(registered_name, cls)

    def decorator(cls: C) -> C:
        register_component(featurizers, "featurizer", name, cls, metadata, validate)
        cls.name = name
        if not getattr(cls, "storage_key", None):
            cls.storage_key = name
        return cls

    return decorator


def get_featurizer(name: str, side: Side | None = None) -> type[Any]:
    """Return the featurizer class registered under *name*, checking that it supports *side*.

    An unknown *name* raises :class:`~drevalpy.errors.UnknownNameError`.

    :param name: Registry name of the featurizer.
    :param side: Side the featurizer is used on; ``None`` skips the side check.
    :returns: Featurizer class registered under *name*.
    """
    cls = featurizers.get(name)
    if side is not None:
        check_side(name, cls, side)
    return cls


def check_side(name: str, cls: type[Any], side: str) -> None:
    """Raise when *cls* does not declare *side* among its ``sides``.

    :param name: Registry name of the featurizer, for the message.
    :param cls: Featurizer class.
    :param side: Side the featurizer is used on.
    :raises ValueError: If *side* is not ``"sample"`` or ``"drug"``, or *cls* does not support it.
    """
    if side not in _SIDES:
        msg = f"Unknown featurizer side {side!r}. Available: drug, sample"
        raise ValueError(msg)
    sides = frozenset(cls.sides)
    if side not in sides:
        msg = (
            f"Featurizer {name!r} cannot be used on the {side} side; "
            f"it supports only: {', '.join(sorted(sides)) or '<none>'}"
        )
        raise ValueError(msg)


def names_for_side(side: Side) -> tuple[str, ...]:
    """Return the registered featurizers that support *side*.

    :param side: ``"sample"`` or ``"drug"``.
    :returns: Names in registration order.
    """
    return tuple(name for name, cls in featurizers.items() if side in cls.sides)
