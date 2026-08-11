"""Comparison scatter plot visualization (Plotly + MultiQC scatter)."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from drevalpy.visualization.base import Section, Visualization
from drevalpy.visualization.registry import visualization_registry
from drevalpy.visualization.requirements import PlotRequirement

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult


def _collect_model_predictions(result: ExperimentResult) -> dict[str, dict[int, dict[int, float]]]:
    """Collect predictions indexed by model name, fold, and sample index."""
    model_runs: dict[str, dict[int, dict[int, float]]] = {}
    for model in result.models:
        fold_data: dict[int, dict[int, float]] = {}
        for run in model.runs:
            if run.randomization is not None:
                continue
            preds = {i: float(run.predictions[i]) for i in range(len(run.predictions))}
            fold_data[run.fold_index] = preds
        model_runs[model.model_name] = fold_data
    return model_runs


def _compute_pair_points(
    model_runs: dict[str, dict[int, dict[int, float]]], name_a: str, name_b: str
) -> list[dict[str, float]]:
    """Compute scatter points for a single model pair."""
    folds_a = model_runs.get(name_a, {})
    folds_b = model_runs.get(name_b, {})
    common_folds = sorted(set(folds_a) & set(folds_b))
    points: list[dict[str, float]] = []
    for fold_idx in common_folds:
        preds_a = folds_a[fold_idx]
        preds_b = folds_b[fold_idx]
        common_indices = sorted(set(preds_a) & set(preds_b))
        for idx in common_indices:
            a_val = preds_a[idx]
            b_val = preds_b[idx]
            if not (np.isnan(a_val) or np.isnan(b_val)):
                points.append({"x": a_val, "y": b_val})
    return points


@visualization_registry.register(
    "comparison_scatter",
    "Pairwise scatter comparing predictions between models",
    requirements=frozenset({PlotRequirement.MULTIPLE_MODELS}),
)
class ComparisonScatterVisualization(Visualization):
    """Pairwise model prediction comparison scatter plots (Plotly)."""

    def __init__(self) -> None:
        """Initialize with empty state."""
        self._fig: go.Figure | None = None
        self._result: ExperimentResult | None = None
        self._pair_data: list[tuple[str, str, list[dict[str, float]]]] = []

    def compute(self, result: ExperimentResult, dataset=None) -> None:
        """Build pairwise scatter figure comparing model predictions.

        :param result: Experiment result with at least two models.
        """
        self._result = result
        self._pair_data = []
        self._fig = go.Figure()

        model_names = [m.model_name for m in result.models]
        if len(model_names) < 2:
            return

        model_runs = _collect_model_predictions(result)
        pairs = list(combinations(model_names, 2))
        first_pair = True

        for name_a, name_b in pairs:
            points = _compute_pair_points(model_runs, name_a, name_b)
            if not points:
                continue
            self._pair_data.append((name_a, name_b, points))

            if first_pair:
                self._fig.add_trace(
                    go.Scatter(
                        x=[p["x"] for p in points],
                        y=[p["y"] for p in points],
                        mode="markers",
                        marker={"size": 4, "opacity": 0.6},
                        name=f"{name_a} vs {name_b}",
                    )
                )
                first_pair = False

        self._fig.update_layout(
            title="Pairwise Model Prediction Comparison",
            xaxis_title=pairs[0][0] if pairs else "",
            yaxis_title=pairs[0][1] if pairs else "",
            showlegend=False,
        )

    def to_png(self, path: str | Path) -> None:
        """Render comparison scatter to a static PNG.

        :param path: Output file path.
        """
        if self._fig is None:
            raise RuntimeError("Call compute() before to_png()")
        self._fig.write_image(str(path))

    def to_multiqc(self) -> list[Section]:
        """Return MultiQC scatter Sections, one per model pair."""
        if self._result is None:
            raise RuntimeError("Call compute() before to_multiqc()")
        try:
            from multiqc.plots import scatter as mqc_scatter
        except ImportError as e:
            raise ImportError("multiqc is required for to_multiqc(). Install with: pip install drevalpy[report]") from e

        sections: list[Section] = []
        for name_a, name_b, points in self._pair_data:
            if not points:
                continue
            pair_name = f"{name_a}_vs_{name_b}"
            datasets = [{pair_name: points}]
            plot = mqc_scatter.plot(
                datasets,
                pconfig={
                    "id": f"dreval_comp_{pair_name}",
                    "title": f"Comparison: {name_a} vs {name_b}",
                    "xlab": name_a,
                    "ylab": name_b,
                },
            )
            sections.append(
                Section(
                    name=f"Comparison: {name_a} vs {name_b}",
                    anchor=f"dreval_comp_{pair_name}",
                    description=f"Pairwise prediction comparison between {name_a} and {name_b}.",
                    plot=plot,
                )
            )

        return sections

    def show(self) -> None:
        """Display the comparison scatter in a Jupyter notebook."""
        if self._fig is None:
            raise RuntimeError("Call compute() before show()")
        self._fig.show()
