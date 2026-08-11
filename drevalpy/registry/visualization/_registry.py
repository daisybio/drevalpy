"""Visualization registry: maps names to visualization classes.

Register custom visualizations with the ``@visualization_registry.register`` decorator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult


class VisualizationRegistry:
    """Registry mapping names to Visualization classes with requirement metadata."""

    def __init__(self) -> None:
        """Initialize with an empty registry."""
        self._store: dict[str, type[Any]] = {}
        self._descriptions: dict[str, str] = {}
        self._requirements: dict[str, frozenset[Any]] = {}
        self._result_types: dict[str, str] = {}

    @property
    def names(self) -> list[str]:
        """Sorted list of registered visualization names."""
        return sorted(self._store)

    def list_names(self) -> list[str]:
        """Alias for names property, for API consistency with other registries."""
        return self.names

    def register(
        self,
        name: str,
        description: str = "",
        *,
        result_type: str = "ExperimentResult",
        requirements: frozenset[Any] = frozenset(),
    ) -> Callable[[type[Any]], type[Any]]:
        """Decorator to register a visualization class.

        :param name: Unique name for this visualization.
        :param description: Human-readable description.
        :param result_type: "ExperimentResult" or "ModelResult".
        :param requirements: frozenset of PlotRequirement values.
        :returns: Class decorator.
        """

        def decorator(cls: type[Any]) -> type[Any]:
            if name in self._store:
                raise ValueError(f"Visualization {name!r} already registered")
            cls.registry_name = name
            self._store[name] = cls
            self._descriptions[name] = description
            self._requirements[name] = requirements
            self._result_types[name] = result_type
            return cls

        return decorator

    def get(self, name: str) -> type[Any]:
        """Return the visualization class registered under name.

        :raises ValueError: If name is not registered.
        """
        if name not in self._store:
            raise ValueError(f"Unknown visualization {name!r}. Registered: {self.names}")
        return self._store[name]

    def applicable(self, experiment: ExperimentResult) -> list[type[Any]]:
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

    def retain_only(self, names: frozenset[str]) -> None:
        """Remove all entries not in the given set (for rollback support).

        :param names: Set of visualization names to keep.
        """
        for name in list(self._store):
            if name not in names:
                del self._store[name]
                self._descriptions.pop(name, None)
                self._requirements.pop(name, None)
                self._result_types.pop(name, None)

    def __repr__(self) -> str:
        """Return a readable string representation."""
        lines = ["VisualizationRegistry:"]
        for name in self.names:
            lines.append(f"  {name}: {self._descriptions.get(name, '')}")
        return "\n".join(lines)


visualization_registry = VisualizationRegistry()
