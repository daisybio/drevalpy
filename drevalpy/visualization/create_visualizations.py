"""Orchestrator that selects and runs applicable plots for a given result."""

from __future__ import annotations

from typing import TYPE_CHECKING

from upath import UPath as Path

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult, ModelResult, RunResult


def create_visualizations(
    result: ExperimentResult | ModelResult | RunResult,
    output_dir: str | Path,
) -> None:
    """Select and run all applicable plots for the given result.

    :param result: An ExperimentResult, ModelResult, or RunResult.
    :param output_dir: Directory to write plot artifacts into.
    """
    from drevalpy.types.results import ExperimentResult, ModelResult, RunResult

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if isinstance(result, RunResult):
        experiment = ExperimentResult([result])
    elif isinstance(result, ModelResult):
        all_runs = list(result.runs)
        experiment = ExperimentResult(all_runs)
    else:
        experiment = result

    plot_classes: list[type] = _get_plot_registry()

    for plot_cls in plot_classes:
        if not experiment.satisfies(plot_cls.requirements):
            continue

        if plot_cls.result_type == "ModelResult":
            for model in experiment.models:
                plot = plot_cls(model)
                plot.draw_and_save(out_prefix=out, out_suffix=f"{experiment.split_mode}_{model.model_name}")
        else:
            plot = plot_cls(experiment)
            plot.draw_and_save(out_prefix=out, out_suffix=experiment.split_mode)


def _get_plot_registry() -> list[type]:
    """Return all registered plot classes.

    Each class declares ``result_type`` and ``requirements`` to support
    automatic selection via :func:`create_visualizations`.
    """
    from drevalpy.visualization.comp_scatter import ComparisonScatter
    from drevalpy.visualization.critical_difference_plot import CriticalDifferencePlot
    from drevalpy.visualization.cross_study_tables import CrossStudyTables
    from drevalpy.visualization.heatmap import Heatmap
    from drevalpy.visualization.regression_slider_plot import RegressionSliderPlot
    from drevalpy.visualization.violin import Violin

    return [
        Violin,
        Heatmap,
        CriticalDifferencePlot,
        RegressionSliderPlot,
        ComparisonScatter,
        CrossStudyTables,
    ]
