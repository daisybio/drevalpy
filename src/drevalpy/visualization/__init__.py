"""Visualization registry, the plot interface, and plot implementations."""

__all__ = [
    "Plot",
    "PlotRequirement",
    "Section",
    "create_report",
    "save_all_png",
    "save_png",
    "show",
    "visualization_registry",
]

from drevalpy.registry.visualization import visualization_registry

from .base import Plot, Section, save_png, show
from .report import create_report, save_all_png
from .requirements import PlotRequirement
