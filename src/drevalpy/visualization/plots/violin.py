"""Violin plot visualization (Plotly + MultiQC violin)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from drevalpy.evaluation import Metric
from drevalpy.log import get_logger
from drevalpy.registry.visualization import register
from drevalpy.visualization.base import Plot, Section
from drevalpy.visualization.plots._utils import runs_frame
from drevalpy.visualization.requirements import PlotRequirement

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from drevalpy.types.results import ExperimentResult

logger = get_logger(__name__)


def _is_finite(value: float) -> bool:
    """Whether *value* is a real number MultiQC can plot.

    :param value: Metric value, possibly NaN or non-numeric.
    :returns: True if the value is finite.
    """
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _run_metrics(result: ExperimentResult) -> dict[str, dict[str, float]]:
    """Collect the metrics of every non-randomized run, keyed by run identity.

    :param result: Experiment to read.
    :returns: ``{run_id: {metric: value}}``, the payload of MultiQC's violin plot.
    """
    return {run.run_id: dict(run.metrics) for model in result.models for run in model.select_runs()}


@register("violin", "Violin plots of evaluation metrics across CV folds")
class ViolinPlot(Plot):
    """Violin plot showing metric distributions across folds per model."""

    requirements = frozenset({PlotRequirement.MULTIPLE_FOLDS})

    def figure(self, result: ExperimentResult) -> go.Figure:
        """Draw one violin per model and metric from the per-fold metrics.

        :param result: Experiment result with multiple folds.
        :returns: The Plotly figure.
        """
        import plotly.graph_objects as go

        df = runs_frame(result).sort_index()
        df["box"] = df["algorithm"] + "_" + df["rand_setting"] + "_" + df["test_mode"]
        df = df.dropna(axis=1, how="all")

        metrics = [m.value for m in Metric if m in df.columns]
        if not metrics:
            logger.warning("violin: no metric has a finite value in any fold; the figure is empty")

        fig = go.Figure()
        for metric in metrics:
            for box in df["box"].unique():
                tmp_df = df[df["box"] == box]
                label = box.split("_")[0] + ": " + metric
                fig.add_trace(
                    go.Violin(
                        y=tmp_df[metric],
                        x=[label] * len(tmp_df[metric]),
                        name=label,
                        box_visible=True,
                        meanline_visible=True,
                    )
                )

        fig.update_layout(title_text="All Metrics", height=600, width=1100)
        return fig

    def sections(self, result: ExperimentResult) -> list[Section]:
        """Return a MultiQC violin Section using the native violin plot API.

        :param result: Experiment result with multiple folds.
        :returns: One section, or none when no metric has a finite value.
        """
        from multiqc.plots import violin as mqc_violin

        data = _run_metrics(result)
        metric_names = sorted({m for metrics in data.values() for m in metrics if _is_finite(metrics[m])})
        if not metric_names:
            logger.warning("violin: no metric has a finite value in any fold; skipping the section")
            return []
        headers: dict[str, dict[str, str]] = {m: {"title": m, "description": f"Metric: {m}"} for m in metric_names}
        plot = mqc_violin.plot(data, headers, pconfig={"id": "dreval_violin"})

        return [
            Section(
                name="Metric Distributions",
                anchor="dreval_violin",
                description="Distribution of evaluation metrics across cross-validation folds.",
                plot=plot,
            )
        ]
