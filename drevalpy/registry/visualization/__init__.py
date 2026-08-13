"""Visualization registry: register, discover, and retrieve visualization classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from ._registry import VisualizationRegistry, visualization_registry

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult
    from drevalpy.visualization.base import Visualization

__all__ = [
    "VisualizationRegistry",
    "applicable",
    "get",
    "list",
    "metadata",
    "register",
    "table",
    "visualization_registry",
]


def register(
    name: str,
    description: str = "",
    *,
    result_type: str = "ExperimentResult",
    requirements=frozenset(),
    override: bool = False,
):
    """Decorator to register a visualization class.

    :param name: Unique name for this visualization.
    :param description: Human-readable description.
    :param result_type: ``"ExperimentResult"`` or ``"ModelResult"``.
    :param requirements: frozenset of ``PlotRequirement`` values.
    :param override: Replace an already-registered name instead of raising.
    :returns: Class decorator.
    """
    return visualization_registry.register(
        name,
        description,
        result_type=result_type,
        requirements=requirements,
        override=override,
    )


def get(name: str) -> type[Visualization]:
    """Return the visualization class registered under name."""
    return visualization_registry.get(name)


def list() -> list[str]:  # noqa: A001
    """Return sorted list of registered visualization names."""
    return visualization_registry.names


def table() -> pd.DataFrame:
    """Return registry contents as a DataFrame.

    :returns: DataFrame with Name, Description, Result type and Requirements columns.
    """
    return visualization_registry.to_dataframe()


def metadata(name: str) -> dict[str, Any]:
    """Return metadata for a registered visualization.

    :param name: Registry name of the visualization.
    :returns: Metadata dict including result type and plot requirements.
    """
    return visualization_registry.get_metadata(name)


def applicable(experiment: ExperimentResult) -> list[type[Visualization]]:
    """Return all visualization classes whose requirements are satisfied."""
    return visualization_registry.applicable(experiment)
