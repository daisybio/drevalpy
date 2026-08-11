"""Visualization registry: register, discover, and retrieve visualization classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._registry import VisualizationRegistry, visualization_registry

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult
    from drevalpy.visualization.base import Visualization

__all__ = [
    "VisualizationRegistry",
    "applicable",
    "get",
    "list",
    "register",
    "table",
    "visualization_registry",
]


def register(name: str, description: str = "", *, result_type: str = "ExperimentResult", requirements=frozenset()):
    """Decorator to register a visualization class."""
    return visualization_registry.register(name, description, result_type=result_type, requirements=requirements)


def get(name: str) -> type[Visualization]:
    """Return the visualization class registered under name."""
    return visualization_registry.get(name)


def list() -> list[str]:  # noqa: A001
    """Return sorted list of registered visualization names."""
    return visualization_registry.names


def table() -> str:
    """Return registry contents as a string representation."""
    return repr(visualization_registry)


def applicable(experiment: ExperimentResult) -> list[type[Visualization]]:
    """Return all visualization classes whose requirements are satisfied."""
    return visualization_registry.applicable(experiment)
