"""Regression scatter plot visualization (MultiQC)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.visualization.base import Section, Visualization
from drevalpy.visualization.registry import visualization_registry

if TYPE_CHECKING:
    from drevalpy.types.results import ModelResult


@visualization_registry.register(
    "regression_scatter",
    "Scatter plot of predicted vs. actual drug response values",
    result_type="ModelResult",
)
class RegressionScatterVisualization(Visualization):
    """Predicted vs. ground-truth scatter for a single model."""

    def generate(self, result: ModelResult) -> list[Section]:
        """Build scatter data from model runs.

        :param result: Model result containing predictions and ground truth.
        :returns: Single-element list with the scatter Section.
        """
        try:
            from multiqc.plots import scatter as mqc_scatter
        except ImportError as e:
            raise ImportError(
                "multiqc is required for regression scatter plots. Install with: pip install drevalpy[report]"
            ) from e

        import numpy as np

        datasets: list[dict[str, list[dict[str, float]]]] = []
        for run in result.runs:
            mask = ~np.isnan(run.ground_truth) & ~np.isnan(run.predictions)
            gt = run.ground_truth[mask]
            pred = run.predictions[mask]

            points = [{"x": float(g), "y": float(p)} for g, p in zip(gt, pred, strict=True)]
            datasets.append({f"fold_{run.fold_index}": points})

        plot = mqc_scatter.plot(
            datasets,
            pconfig={
                "id": f"dreval_scatter_{result.model_name}",
                "title": f"Regression Scatter: {result.model_name}",
                "xlab": "Ground Truth",
                "ylab": "Predicted",
            },
        )

        return [
            Section(
                name=f"Regression Scatter: {result.model_name}",
                anchor=f"dreval_scatter_{result.model_name}",
                description=(
                    f"Predicted vs. ground-truth values for {result.model_name} across {result.n_folds} fold(s)."
                ),
                plot=plot,
            )
        ]
