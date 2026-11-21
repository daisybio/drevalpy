"""The featurizer registries, one per side, their ``register`` decorator and the lookup.

A featurizer works on exactly one side, which it names when it registers. Each side has a
registry of its own, so a name is unique within a side: ``oneHot`` exists as a sample
featurizer and as a drug featurizer, two registrations that share their code. A config slot
picks the side, and the lookup reads that side's registry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Any, TypeVar

from drevalpy.components.contracts.hyperparameter_space import validate_component_hyperparameter_space
from drevalpy.errors import UnknownNameError
from drevalpy.registry._component import LiteratureReference, component_decorator
from drevalpy.registry._registry import Registry
from drevalpy.registry.featurizer._validate import validate_featurizer_declarations
from drevalpy.types.data.feature_source import Side
from drevalpy.types.enums.accelerator import Accelerator

C = TypeVar("C", bound=type[Any])

#: The registered featurizer classes of each side, in registration order.
featurizer_registries: Mapping[Side, Registry[type[Any]]] = MappingProxyType(
    {side: Registry(f"{side} featurizer") for side in Side}
)


def _side(side: Side | str) -> Side:
    """Return *side* as a :class:`Side`.

    :param side: A side, or its value.
    :returns: The side.
    :raises ValueError: If *side* is not ``"sample"`` or ``"drug"``.
    """
    try:
        return Side(side)
    except ValueError:
        msg = f"Unknown featurizer side {side!r}. Available: drug, sample"
        raise ValueError(msg) from None


def register(
    name: str,
    *,
    side: Side,
    description: str,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
    accelerator: Accelerator | str = Accelerator.NONE,
    license: str | None = None,
    artifact_license: str | None = None,
) -> Callable[[C], C]:
    """Return a class decorator that registers a featurizer under *name* on *side*.

    The decorator sets ``name``, ``side``, ``accelerator``, ``license`` and
    ``artifact_license`` on the class. To offer the same featurizer on both sides, register one
    subclass per side.

    :param name: Registry name used in model configs and listings, unique within *side*.
    :param side: The side the featurizer reads.
    :param description: Short human-readable summary.
    :param tags: Optional discovery tags.
    :param reference: Optional literature citation metadata.
    :param accelerator: ``"optional"`` when the featurizer uses a torch accelerator
        if there is one, ``"required"`` when it cannot run without one.
    :param license: SPDX license of the code the featurizer was ported from; required with a
        *reference*. Without one, the featurizer has the license of the package that ships it.
    :param artifact_license: SPDX license of the third-party files the featurizer downloads.
    :returns: Class decorator that registers and returns the decorated class.
    """
    resolved = _side(side)

    def validate(registered_name: str, cls: type[Any]) -> None:
        validate_component_hyperparameter_space(registered_name, cls)
        validate_featurizer_declarations(registered_name, cls)

    register_class = component_decorator(
        featurizer_registries[resolved],
        name,
        description=description,
        tags=tags,
        reference=reference,
        accelerator=accelerator,
        license=license,
        artifact_license=artifact_license,
        validate=validate,
    )

    def decorator(cls: C) -> C:
        register_class(cls)
        cls.name = name
        cls.side = resolved
        return cls

    return decorator


def featurizer_registry(side: Side | str) -> Registry[type[Any]]:
    """Return the registry of the featurizers of *side*.

    :param side: ``Side.SAMPLE`` or ``Side.DRUG``.
    :returns: That side's registry.
    """
    return featurizer_registries[_side(side)]


def get_featurizer(name: str, side: Side | str) -> type[Any]:
    """Return the featurizer class registered under *name* on *side*.

    An unknown *name* raises :class:`~drevalpy.errors.UnknownNameError`, which says so when
    *name* is a featurizer of the other side.

    :param name: Registry name of the featurizer.
    :param side: Side the featurizer is used on.
    :returns: Featurizer class registered under *name* on *side*.
    :raises UnknownNameError: If *name* is not registered on *side*.
    """
    resolved = _side(side)
    registry = featurizer_registries[resolved]
    if name not in registry:
        other = Side.DRUG if resolved is Side.SAMPLE else Side.SAMPLE
        if name in featurizer_registries[other]:
            msg = f"{name!r} is a {other} featurizer and cannot be used on the {resolved} side"
            raise UnknownNameError(msg)
    return registry.get(name)


def names_for_side(side: Side | str) -> tuple[str, ...]:
    """Return the names of the featurizers registered on *side*.

    :param side: ``Side.SAMPLE`` or ``Side.DRUG``.
    :returns: Names in registration order.
    """
    return featurizer_registry(side).names()
