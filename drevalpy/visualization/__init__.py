"""Report plots and leaderboard helpers."""

__all__ = [
    "ComparisonScatter",
    "CriticalDifferencePlot",
    "CrossStudyTables",
    "Heatmap",
    "ImageVisualization",
    "PlotRequirement",
    "RegressionSliderPlot",
    "Section",
    "VioHeat",
    "Violin",
    "Visualization",
    "create_report",
    "create_visualizations",
    "save_all_png",
    "visualization_registry",
]

from ._legacy import (  # noqa: E402 — backward compatibility re-exports
    ComparisonScatter,
    CriticalDifferencePlot,
    CrossStudyTables,
    Heatmap,
    RegressionSliderPlot,
    VioHeat,
    Violin,
    create_visualizations,
)
from .base import ImageVisualization, Section, Visualization
from .registry import visualization_registry
from .report import create_report, save_all_png
from .requirements import PlotRequirement
