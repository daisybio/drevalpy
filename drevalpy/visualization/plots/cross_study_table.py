"""Cross-study table visualization (MultiQC)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.visualization.base import Section, Visualization
from drevalpy.visualization.registry import visualization_registry

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult


@visualization_registry.register(
    "cross_study_table",
    "Summary table of model metrics",
)
class CrossStudyTableVisualization(Visualization):
    """Tabular summary of model performance metrics."""

    def generate(self, result: ExperimentResult) -> list[Section]:
        """Build a table of aggregate metrics per model.

        :param result: Experiment result to summarize.
        :returns: Single-element list with the table Section.
        """
        try:
            from multiqc.plots import table as mqc_table
        except ImportError as e:
            raise ImportError(
                "multiqc is required for cross-study tables. Install with: pip install drevalpy[report]"
            ) from e

        metric_names = sorted({m for model in result.models for m in model.aggregate_metrics})

        table_data: dict[str, dict[str, float]] = {}
        for model in result.models:
            row: dict[str, float] = {}
            for metric in metric_names:
                agg = model.aggregate_metrics.get(metric)
                if agg:
                    row[f"{metric}_mean"] = agg["mean"]
                    row[f"{metric}_std"] = agg["std"]
            table_data[model.model_name] = row

        headers: dict[str, dict[str, str]] = {}
        for metric in metric_names:
            headers[f"{metric}_mean"] = {
                "title": f"{metric} (mean)",
                "description": f"Mean {metric} across folds",
                "format": "{:,.4f}",
            }
            headers[f"{metric}_std"] = {
                "title": f"{metric} (std)",
                "description": f"Std of {metric} across folds",
                "format": "{:,.4f}",
            }

        plot = mqc_table.plot(
            table_data,
            headers,
            pconfig={"id": "dreval_summary_table", "title": "Model Summary"},
        )

        return [
            Section(
                name="Model Summary Table",
                anchor="dreval_summary_table",
                description="Aggregate performance metrics (mean ± std) per model.",
                plot=plot,
            )
        ]
