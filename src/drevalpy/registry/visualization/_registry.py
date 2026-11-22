"""The visualization registry, its ``register`` decorator and ``applicable``."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from drevalpy.registry._registry import Registry

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult
    from drevalpy.visualization.base import Plot


P = TypeVar("P", bound="Plot")

#: Every registered plot class, in registration order.
visualization_registry: Registry[type[Plot]] = Registry("visualization")


def register(name: str, description: str = "", *, replace: bool = False) -> Callable[[type[P]], type[P]]:
    """Return a class decorator that registers a plot class under *name* and sets its ``name``.

    The registry metadata (``description``, ``result_type``, ``requirements``) is read from the
    arguments and the class, for the listing table. Applying the decorator raises ``ValueError``
    if *name* is already registered and *replace* is false.

    :param name: Unique name for this plot.
    :param description: Human-readable description.
    :param replace: Replace an already-registered name instead of raising.
    :returns: Class decorator that registers the decorated class under *name*.
    """

    def decorator(cls: type[P]) -> type[P]:
        from drevalpy.visualization.base import Plot

        if not (isinstance(cls, type) and issubclass(cls, Plot)):
            msg = f"visualization {name!r} must be a subclass of drevalpy.visualization.base.Plot, got {cls!r}"
            raise TypeError(msg)
        metadata = {
            "description": description,
            "result_type": cls.result_type.__name__,
            "requirements": frozenset(cls.requirements),
        }
        visualization_registry.register(name, cls, metadata=metadata, replace=replace)
        cls.name = name
        return cls

    return decorator


def applicable(experiment: ExperimentResult) -> list[type[Plot]]:
    """Return the plot classes whose requirements *experiment* satisfies.

    :param experiment: The experiment result to check against.
    :returns: Applicable plot classes, in registration order.
    """
    from drevalpy.visualization.requirements import requirements_met

    return [cls for _, cls in visualization_registry.items() if requirements_met(experiment, cls.requirements)]
