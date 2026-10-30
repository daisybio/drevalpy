"""Tests for the contract every built-in plot shares: it is registered, declares its result type and keeps no state."""

from __future__ import annotations

import pytest

from drevalpy.registry.visualization import visualization_registry
from drevalpy.types.results import ExperimentResult, ModelResult
from drevalpy.visualization.base import Plot, Section
from drevalpy.visualization.plots.comparison_scatter import ComparisonScatterPlot
from drevalpy.visualization.plots.critical_difference import CriticalDifferencePlot
from drevalpy.visualization.plots.heatmap import HeatmapPlot
from drevalpy.visualization.plots.leaderboard import LeaderboardPlot
from drevalpy.visualization.plots.regression_scatter import RegressionScatterPlot
from drevalpy.visualization.plots.violin import ViolinPlot

BUILTIN_PLOTS = {
    "comparison_scatter": ComparisonScatterPlot,
    "critical_difference": CriticalDifferencePlot,
    "heatmap": HeatmapPlot,
    "leaderboard": LeaderboardPlot,
    "regression_scatter": RegressionScatterPlot,
    "violin": ViolinPlot,
}


@pytest.mark.parametrize(("name", "plot_class"), BUILTIN_PLOTS.items())
def test_each_builtin_is_registered_under_its_name_with_metadata_read_from_the_class(name, plot_class) -> None:
    metadata = visualization_registry.metadata(name)

    assert visualization_registry.get(name) is plot_class
    assert issubclass(plot_class, Plot)
    assert plot_class.name == name
    assert plot_class.result_type in (ExperimentResult, ModelResult)
    assert metadata["result_type"] == plot_class.result_type.__name__
    assert metadata["requirements"] == plot_class.requirements


class TestStateless:
    pytestmark = pytest.mark.slow

    @pytest.mark.parametrize("plot_class", BUILTIN_PLOTS.values(), ids=BUILTIN_PLOTS)
    def test_drawing_leaves_the_instance_unchanged(self, plot_class, experiment) -> None:
        plot = plot_class()
        before = dict(vars(plot))
        result = experiment.models[0] if plot_class.result_type is ModelResult else experiment

        sections = plot.sections(result)
        plot.figure(result)

        assert vars(plot) == before
        assert sections
        assert all(isinstance(section, Section) for section in sections)
