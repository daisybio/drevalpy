"""Bind one side-agnostic featurizer implementation to both entity sides.

``Featurizer.side`` is a ``ClassVar`` stamped onto the class by the registry
(``FeaturizerRegistry.register``), and ``list_stored_variants`` is a
``classmethod`` reading ``cls.side``. A single class therefore cannot be
registered on both sides - the second registration would overwrite the first
one's ``side``. :func:`register_for_sides` resolves that by deriving one subclass
per side from the shared implementation, so each registry gets its own class
object to stamp.

The derived classes are injected back into the defining module's namespace
because ``_reregister_from_module`` in ``drevalpy/registry/_builtins.py`` walks
``vars(module)`` and dispatches on each class's ``side``; a class living only in
this decorator's closure would silently vanish from the registries after a
registry ``clear()``.

:func:`register_for_sides` is **public**, re-exported from
:mod:`drevalpy.plugin` and covered by that facade's compatibility promise. The
module keeps its leading underscore anyway: ``_discover_modules`` in
``drevalpy/registry/_builtins.py`` skips ``_``-prefixed files, and this module
must stay out of the component scan. So the *module path* is private and may
move; the symbol reached through ``drevalpy.plugin`` may not.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from typing import Any

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.types.enums.literature_reference import LiteratureReference

#: Entity side -> (derived-class name prefix, ``module:ClassName`` of the side base).
_SIDE_BASES: dict[str, tuple[str, str]] = {
    "cell_line": ("CellLine", "drevalpy.components.featurizers.cell_line.base:CellLineFeaturizer"),
    "drug": ("Drug", "drevalpy.components.featurizers.drug.base:DrugFeaturizer"),
}

#: Marker stripped out of a shared implementation's name to make room for the side
#: prefix: ``SharedIdentityFeaturizer`` -> ``CellLineIdentityFeaturizer``.
_SHARED_PREFIX = "Shared"


def known_sides() -> tuple[str, ...]:
    """Return the entity sides a shared featurizer can be bound to.

    :returns: Sorted side names.
    """
    return tuple(sorted(_SIDE_BASES))


def _side_base(side: str) -> type[Any]:
    """Import and return the featurizer base class for *side*.

    :param side: Entity side, ``"cell_line"`` or ``"drug"``.
    :returns: That side's ``Featurizer`` subclass.
    :raises ValueError: If *side* is not a known entity side.
    """
    entry = _SIDE_BASES.get(side)
    if entry is None:
        msg = f"unknown featurizer side {side!r}; expected one of {list(known_sides())}"
        raise ValueError(msg)
    module_name, class_name = entry[1].split(":")
    return getattr(__import__(module_name, fromlist=[class_name]), class_name)


def _side_register(side: str) -> Callable[..., Callable[[type[Any]], type[Any]]]:
    """Return the ``register`` decorator factory of the *side* registry.

    :param side: Entity side, ``"cell_line"`` or ``"drug"``.
    :returns: That registry's ``register`` function.
    """
    return __import__(f"drevalpy.registry.{side}_featurizer", fromlist=["register"]).register


def derived_class_name(implementation_name: str, side: str) -> str:
    """Return the class name for the *side* binding of an implementation.

    :param implementation_name: ``__name__`` of the shared implementation class.
    :param side: Entity side, ``"cell_line"`` or ``"drug"``.
    :returns: Side-prefixed class name.
    :raises ValueError: If *side* is not a known entity side.
    """
    entry = _SIDE_BASES.get(side)
    if entry is None:
        msg = f"unknown featurizer side {side!r}; expected one of {list(known_sides())}"
        raise ValueError(msg)
    return f"{entry[0]}{implementation_name.removeprefix(_SHARED_PREFIX)}"


def _derive(implementation: type[Any], side: str) -> type[Any]:
    """Create the *side*-bound subclass of *implementation*.

    The side base carries ``ABCMeta``, so the subclass is built through that
    metaclass rather than through ``type`` directly.

    :param implementation: Side-agnostic implementation class.
    :param side: Entity side, ``"cell_line"`` or ``"drug"``.
    :returns: Freshly created, not-yet-registered subclass.
    """
    base = _side_base(side)
    return type(base)(
        derived_class_name(implementation.__name__, side),
        (implementation, base),
        {"__module__": implementation.__module__, "__doc__": implementation.__doc__},
    )


def register_for_sides(
    name: str,
    *,
    description: str | dict[str, str],
    contract: FeatureContract | FeatureFormat | None = None,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
    sides: Iterable[str] = ("cell_line", "drug"),
) -> Callable[[type[Any]], type[Any]]:
    """Register one side-agnostic implementation under *name* on every side.

    For each side a subclass of the decorated implementation is derived against
    that side's featurizer base, registered under *name*, and bound into the
    defining module's namespace under a side-prefixed class name. The decorated
    implementation is returned unregistered, so it stays importable as the shared
    logic it is.

    :param name: Registry name, identical on every side.
    :param description: Registry description; pass a ``{side: text}`` mapping to
        word it per side, or a single string used for every side.
    :param contract: Feature format contract, forwarded to each registration.
    :param tags: Optional discovery tags, forwarded to each registration.
    :param reference: Optional literature citation, forwarded to each registration.
    :param sides: Entity sides to bind; both by default.
    :returns: Class decorator returning the undecorated implementation.
    """

    def decorator(implementation: type[Any]) -> type[Any]:
        module = sys.modules[implementation.__module__]
        for side in sides:
            derived = _derive(implementation, side)
            text = description[side] if isinstance(description, dict) else description
            registered = _side_register(side)(
                name,
                description=text,
                contract=contract,
                tags=tags,
                reference=reference,
            )(derived)
            setattr(module, registered.__name__, registered)
        return implementation

    return decorator
