"""Violin plot visualization (Plotly + MultiQC violin).

One violin is the distribution of one metric over the folds of one model, in both outputs;
:func:`_violins` collects them once, so the figure and the report section cannot disagree.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from drevalpy.evaluation import Metric
from drevalpy.log import get_logger
from drevalpy.registry.visualization import register
from drevalpy.visualization.base import Plot, Section
from drevalpy.visualization.requirements import PlotRequirement

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from drevalpy.types.results import ExperimentResult

logger = get_logger(__name__)


def _violins(result: ExperimentResult) -> dict[tuple[str, str], dict[str, float]]:
    """Collect the distribution behind each violin: one per model and metric.

    Only non-randomized runs count, and only finite values: a model that never reports a
    metric gets no violin for it. Both outputs draw exactly these violins.

    :param result: Experiment to read.
    :returns: ``{(model, metric): {run_id: value}}``, metric-major in :class:`Metric` order.
    """
    violins: dict[tuple[str, str], dict[str, float]] = {}
    for metric in Metric:
        for model in result.models:
            values = {
                run.run_id: run.metrics[metric.value]
                for run in model.select_runs()
                if math.isfinite(run.metrics.get(metric.value, math.nan))
            }
            if values:
                violins[(model.model_name, metric.value)] = values
    return violins


@register("violin", "Violin plots of evaluation metrics across CV folds")
class ViolinPlot(Plot):
    """Violin plot showing one metric distribution across folds per model."""

    requirements = frozenset({PlotRequirement.MULTIPLE_FOLDS})

    def figure(self, result: ExperimentResult) -> go.Figure:
        """Draw one violin per model and metric from the per-fold metrics.

        :param result: Experiment result with multiple folds.
        :returns: The Plotly figure.
        """
        import plotly.graph_objects as go

        violins = _violins(result)
        if not violins:
            logger.warning("violin: no metric has a finite value in any fold; the figure is empty")

        fig = go.Figure()
        for (model, metric), values in violins.items():
            label = f"{model}: {metric}"
            fig.add_trace(
                go.Violin(
                    y=list(values.values()),
                    x=[label] * len(values),
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

        violins = _violins(result)
        if not violins:
            logger.warning("violin: no metric has a finite value in any fold; skipping the section")
            return []
        # MultiQC draws one violin per column, with the samples as its points: a column per
        # model and metric, filled only on the rows of that model's runs.
        data: dict[str, dict[str, float]] = {}
        headers: dict[str, dict[str, str]] = {}
        for (model, metric), values in violins.items():
            column = f"{model}: {metric}"
            headers[column] = {"title": metric, "namespace": model, "description": f"{metric} of {model}"}
            for run_id, value in values.items():
                data.setdefault(run_id, {})[column] = value
        plot = mqc_violin.plot(data, headers, pconfig={"id": "dreval_violin", "title": "Metric Distributions"})

        return [
            Section(
                name="Metric Distributions",
                anchor="dreval_violin",
                description="Distribution of evaluation metrics across folds, one violin per model and metric.",
                plot=plot,
            )
        ]
