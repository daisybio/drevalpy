"""Module containing the drevalpy plotly visualizations."""

__all__ = [
    "CorrelationComparisonScatter",
    "CriticalDifferencePlot",
    "Heatmap",
    "HTMLTable",
    "RegressionSliderPlot",
    "VioHeat",
    "Violin",
]

from .corr_comp_scatter import CorrelationComparisonScatter
from .critical_difference_plot import CriticalDifferencePlot
from .heatmap import Heatmap
from .html_tables import HTMLTable
from .regression_slider_plot import RegressionSliderPlot
from .vioheat import VioHeat
from .violin import Violin
