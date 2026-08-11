"""Heatmap visualization (MultiQC)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.visualization.base import Section, Visualization
from drevalpy.visualization.registry import visualization_registry
from drevalpy.visualization.requirements import PlotRequirement

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult


@visualization_registry.register(
    "heatmap",
    "Heatmap of mean metrics per model",
    requirements=frozenset({PlotRequirement.MULTIPLE_FOLDS}),
)
class HeatmapVisualization(Visualization):
    """Heatmap showing mean metric values (rows=models, cols=metrics)."""

    def generate(self, result: ExperimentResult) -> list[Section]:
        """Build heatmap from aggregate metrics.

        :param result: Experiment result with multiple folds.
        :returns: Single-element list with a heatmap Section.
        """
        try:
            from multiqc.plots import heatmap as mqc_heatmap
        except ImportError as e:
            raise ImportError(
                "multiqc is required for heatmap plots. Install with: pip install drevalpy[report]"
            ) from e

        metric_names = sorted({m for model in result.models for m in model.aggregate_metrics})
        model_names = [m.model_name for m in result.models]

        data: list[list[float | None]] = []
        for model in result.models:
            row: list[float | None] = []
            for metric in metric_names:
                agg = model.aggregate_metrics.get(metric)
                row.append(agg["mean"] if agg else None)
            data.append(row)

        plot = mqc_heatmap.plot(
            data,
            xcats=metric_names,
            ycats=model_names,
            pconfig={
                "id": "dreval_heatmap",
                "title": "Model Performance Heatmap",
                "square": False,
            },
        )

        return [
            Section(
                name="Performance Heatmap",
                anchor="dreval_heatmap",
                description="Mean metric values per model across cross-validation folds.",
                plot=plot,
            )
        ]
