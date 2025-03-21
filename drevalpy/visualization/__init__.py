"""Module containing the drevalpy plotly visualizations."""

__all__ = [
    "CorrelationComparisonScatter",
    "CriticalDifferencePlot",
    "Heatmap",
    "HTMLTable",
    "RegressionSliderPlot",
    "VioHeat",
    "Violin",
    "CrossStudyTables",
]

from .corr_comp_scatter import CorrelationComparisonScatter
from .critical_difference_plot import CriticalDifferencePlot
from .cross_study_tables import CrossStudyTables
from .heatmap import Heatmap
from .html_tables import HTMLTable
from .regression_slider_plot import RegressionSliderPlot
from .vioheat import VioHeat
from .violin import Violin
