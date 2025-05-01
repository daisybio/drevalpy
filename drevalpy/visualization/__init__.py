"""Module containing the drevalpy plotly visualizations."""

__all__ = [
    "ComparisonScatter",
    "CriticalDifferencePlot",
    "Heatmap",
    "RegressionSliderPlot",
    "VioHeat",
    "Violin",
    "CrossStudyTables",
]

from .comp_scatter import ComparisonScatter
from .critical_difference_plot import CriticalDifferencePlot
from .cross_study_tables import CrossStudyTables
from .heatmap import Heatmap
from .regression_slider_plot import RegressionSliderPlot
from .vioheat import VioHeat
from .violin import Violin
