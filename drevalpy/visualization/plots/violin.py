"""Violin plot visualization (MultiQC)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.visualization.base import Section, Visualization
from drevalpy.visualization.registry import visualization_registry
from drevalpy.visualization.requirements import PlotRequirement

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult


@visualization_registry.register(
    "violin",
    "Violin plots of evaluation metrics across CV folds",
    requirements=frozenset({PlotRequirement.MULTIPLE_FOLDS}),
)
class ViolinVisualization(Visualization):
    """Violin plot showing metric distributions across folds per model."""

    def generate(self, result: ExperimentResult) -> list[Section]:
        """Build violin plot data from per-fold metrics.

        :param result: Experiment result with multiple folds.
        :returns: Single-element list with a violin Section.
        """
        try:
            from multiqc.plots import violin as mqc_violin
        except ImportError as e:
            raise ImportError("multiqc is required for violin plots. Install with: pip install drevalpy[report]") from e

        data: dict[str, dict[str, float]] = {}
        for model in result.models:
            for run in model.runs:
                sample_name = f"{model.model_name}_fold{run.fold_index}"
                data[sample_name] = dict(run.metrics)

        metric_names = sorted({m for metrics in data.values() for m in metrics})
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
