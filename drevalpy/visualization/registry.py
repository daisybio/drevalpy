"""Visualization registry: maps names to visualization classes.

Register custom visualizations with the ``@visualization_registry.register`` decorator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from drevalpy.visualization.base import Visualization

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult
    from drevalpy.visualization.requirements import PlotRequirement


class VisualizationRegistry:
    """Registry mapping names to Visualization classes with requirement metadata."""

    def __init__(self) -> None:
        """Initialize with an empty registry."""
        self._store: dict[str, type[Visualization]] = {}
        self._descriptions: dict[str, str] = {}
        self._requirements: dict[str, frozenset[PlotRequirement]] = {}
        self._result_types: dict[str, str] = {}

    @property
    def names(self) -> list[str]:
        """Sorted list of registered visualization names."""
        return sorted(self._store)

    def register(
        self,
        name: str,
        description: str = "",
        *,
        result_type: str = "ExperimentResult",
        requirements: frozenset[PlotRequirement] = frozenset(),
    ) -> Callable[[type[Visualization]], type[Visualization]]:
        """Decorator to register a visualization class.

        :param name: Unique name for this visualization.
        :param description: Human-readable description.
        :param result_type: "ExperimentResult" or "ModelResult".
        :param requirements: frozenset of PlotRequirement values.
        :returns: Class decorator.
        """

        def decorator(cls: type[Visualization]) -> type[Visualization]:
            if name in self._store:
                raise ValueError(f"Visualization {name!r} already registered")
            cls.registry_name = name
            self._store[name] = cls
            self._descriptions[name] = description
            self._requirements[name] = requirements
            self._result_types[name] = result_type
            return cls

        return decorator

    def get(self, name: str) -> type[Visualization]:
        """Return the visualization class registered under name.

        :raises ValueError: If name is not registered.
        """
        if name not in self._store:
            raise ValueError(f"Unknown visualization {name!r}. Registered: {self.names}")
        return self._store[name]

    def applicable(self, experiment: ExperimentResult) -> list[type[Visualization]]:
        """Return all visualization classes whose requirements are satisfied.

        :param experiment: The experiment result to check against.
        :returns: List of applicable visualization classes.
        """
        result = []
        for name, cls in self._store.items():
            reqs = self._requirements[name]
            if experiment.satisfies(reqs):
                result.append(cls)
        return result

    def describe(self, name: str) -> str:
        """Return the description for a registered visualization."""
        return self._descriptions.get(name, "")

    def __repr__(self) -> str:
        """Return a readable string representation."""
        lines = ["VisualizationRegistry:"]
        for name in self.names:
            lines.append(f"  {name}: {self._descriptions.get(name, '')}")
        return "\n".join(lines)


visualization_registry = VisualizationRegistry()
