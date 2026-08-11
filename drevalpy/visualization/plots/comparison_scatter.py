"""Comparison scatter plot visualization (MultiQC)."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from drevalpy.visualization.base import Section, Visualization
from drevalpy.visualization.registry import visualization_registry
from drevalpy.visualization.requirements import PlotRequirement

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult


@visualization_registry.register(
    "comparison_scatter",
    "Pairwise scatter comparing predictions between models",
    requirements=frozenset({PlotRequirement.MULTIPLE_MODELS}),
)
class ComparisonScatterVisualization(Visualization):
    """Pairwise model prediction comparison scatter plots."""

    def generate(self, result: ExperimentResult) -> list[Section]:
        """Build pairwise scatter data between models.

        :param result: Experiment result with at least two models.
        :returns: List of Sections, one per model pair.
        """
        try:
            from multiqc.plots import scatter as mqc_scatter
        except ImportError as e:
            raise ImportError(
                "multiqc is required for comparison scatter plots. Install with: pip install drevalpy[report]"
            ) from e

        import numpy as np

        sections: list[Section] = []

        for model_a, model_b in combinations(result.models, 2):
            folds_a = {r.fold_index: r for r in model_a.runs}
            folds_b = {r.fold_index: r for r in model_b.runs}
            common_folds = sorted(set(folds_a) & set(folds_b))

            if not common_folds:
                continue

            points: list[dict[str, float]] = []
            for fold_idx in common_folds:
                run_a = folds_a[fold_idx]
                run_b = folds_b[fold_idx]
                mask = ~np.isnan(run_a.predictions) & ~np.isnan(run_b.predictions)
                pred_a = run_a.predictions[mask]
                pred_b = run_b.predictions[mask]
                points.extend({"x": float(a), "y": float(b)} for a, b in zip(pred_a, pred_b, strict=True))

            pair_name = f"{model_a.model_name}_vs_{model_b.model_name}"
            datasets = [{pair_name: points}]

            plot = mqc_scatter.plot(
                datasets,
                pconfig={
                    "id": f"dreval_comp_{pair_name}",
                    "title": f"Comparison: {model_a.model_name} vs {model_b.model_name}",
                    "xlab": model_a.model_name,
                    "ylab": model_b.model_name,
                },
            )

            sections.append(
                Section(
                    name=f"Comparison: {model_a.model_name} vs {model_b.model_name}",
                    anchor=f"dreval_comp_{pair_name}",
                    description=(
                        f"Pairwise prediction comparison between {model_a.model_name} and {model_b.model_name}."
                    ),
                    plot=plot,
                )
            )

        return sections
