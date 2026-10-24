"""Tests for the contract every plot shares: its output methods refuse to run before ``compute``."""

from __future__ import annotations

import pytest

from drevalpy.visualization.plots.comparison_scatter import ComparisonScatterVisualization
from drevalpy.visualization.plots.critical_difference import CriticalDifferenceVisualization
from drevalpy.visualization.plots.heatmap import HeatmapVisualization
from drevalpy.visualization.plots.leaderboard import LeaderboardVisualization
from drevalpy.visualization.plots.regression_scatter import RegressionScatterVisualization
from drevalpy.visualization.plots.violin import ViolinVisualization


@pytest.mark.parametrize(
    "plot_class",
    [
        ComparisonScatterVisualization,
        CriticalDifferenceVisualization,
        HeatmapVisualization,
        LeaderboardVisualization,
        RegressionScatterVisualization,
        ViolinVisualization,
    ],
)
@pytest.mark.parametrize("method", ["to_png", "to_multiqc", "show"])
def test_methods_raise_before_compute(plot_class: type, tmp_path, method: str) -> None:
    args = (tmp_path / "plot.png",) if method == "to_png" else ()

    with pytest.raises(RuntimeError, match=rf"Call compute\(\) before {method}\(\)"):
        getattr(plot_class(), method)(*args)
