"""Visualization registry: maps names to visualization classes.

Register custom visualizations with the ``@visualization_registry.register`` decorator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd

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
        override: bool = False,
    ) -> Callable[[type[Any]], type[Any]]:
        """Decorator to register a visualization class.

        :param name: Unique name for this visualization.
        :param description: Human-readable description.
        :param result_type: "ExperimentResult" or "ModelResult".
        :param requirements: frozenset of PlotRequirement values.
        :param override: Replace an already-registered name instead of raising.
        :returns: Class decorator.
        :raises ValueError: If *name* is already registered and *override* is false.
        """

        def decorator(cls: type[Any]) -> type[Any]:
            if name in self._store and not override:
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

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Return the metadata record for the visualization registered under *name*.

        Mirrors ``get_metadata`` on the component registries so every registry can
        be introspected the same way.

        :param name: Registered visualization name.
        :returns: Metadata dict for catalog listings.
        :raises ValueError: If *name* is not registered.
        """
        cls = self.get(name)
        return {
            "registry": "visualizations",
            "name": name,
            "class_name": cls.__name__,
            "description": self._descriptions.get(name, ""),
            "result_type": self._result_types.get(name, ""),
            "requirements": frozenset(self._requirements.get(name, frozenset())),
        }

    def list_metadata(self) -> list[dict[str, Any]]:
        """Return metadata for every registered visualization.

        :returns: List of metadata dicts, ordered by visualization name.
        """
        return [self.get_metadata(name) for name in self.names]

    def to_dataframe(self) -> pd.DataFrame:
        """Return registry contents as a pandas DataFrame.

        :returns: One row per visualization, with Name, Description, Result type
            and Requirements columns.
        """
        rows = []
        for name in self.names:
            meta = self.get_metadata(name)
            rows.append(
                {
                    "Name": name,
                    "Description": meta["description"],
                    "Result type": meta["result_type"],
                    "Requirements": ", ".join(sorted(str(req) for req in meta["requirements"])),
                }
            )
        return pd.DataFrame(rows)

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
        """Return a tabular string representation."""
        return self.to_dataframe().to_string(index=False)

    def _repr_html_(self) -> str:
        """HTML table for Jupyter notebooks."""
        return self.to_dataframe().to_html(index=False)


visualization_registry = VisualizationRegistry()
