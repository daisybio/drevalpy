"""Legacy HTML-based visualization classes (deprecated)."""

from .comp_scatter import ComparisonScatter
from .create_visualizations import create_visualizations
from .critical_difference_plot import CriticalDifferencePlot
from .cross_study_tables import CrossStudyTables
from .heatmap import Heatmap
from .outplot import OutPlot
from .regression_slider_plot import RegressionSliderPlot
from .vioheat import VioHeat
from .violin import Violin

__all__ = [
    "ComparisonScatter",
    "CriticalDifferencePlot",
    "CrossStudyTables",
    "Heatmap",
    "OutPlot",
    "RegressionSliderPlot",
    "VioHeat",
    "Violin",
    "create_visualizations",
]
