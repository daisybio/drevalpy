"""Violin plot visualization (Plotly + MultiQC violin)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

from drevalpy.log import get_logger
from drevalpy.registry.visualization import register
from drevalpy.visualization.base import Section, Visualization
from drevalpy.visualization.requirements import PlotRequirement

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult

logger = get_logger(__name__)

_ALL_METRICS = [
    "R^2",
    "R^2: normalized",
    "Pearson",
    "Pearson: normalized",
    "Spearman",
    "Spearman: normalized",
    "Kendall",
    "Kendall: normalized",
    "MSE",
    "RMSE",
    "MAE",
]


def _is_finite(value: float) -> bool:
    """Whether *value* is a real number MultiQC can plot.

    :param value: Metric value, possibly NaN or non-numeric.
    :returns: True if the value is finite.
    """
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _build_df_from_experiment(result: ExperimentResult) -> pd.DataFrame:
    """Build a flat DataFrame from an ExperimentResult.

    Columns: algorithm, rand_setting, test_mode, CV_split, and one column per metric.
    """
    rows: list[dict] = []
    for model in result.models:
        for run in model.runs:
            row: dict = {
                "algorithm": run.model_name,
                "rand_setting": (
                    f"{run.randomization[0]}_{run.randomization[1]}" if run.randomization else "predictions"
                ),
                "test_mode": result.split_mode,
                "CV_split": run.fold_index,
            }
            row.update(run.metrics)
            rows.append(row)
    return pd.DataFrame(rows)


@register(
    "violin",
    "Violin plots of evaluation metrics across CV folds",
    requirements=frozenset({PlotRequirement.MULTIPLE_FOLDS}),
)
class ViolinVisualization(Visualization):
    """Violin plot showing metric distributions across folds per model."""

    def __init__(self) -> None:
        """Initialize with empty state."""
        self._fig: go.Figure | None = None
        self._data: dict[str, dict[str, float]] | None = None

    def compute(self, result: ExperimentResult, dataset=None) -> None:
        """Build violin plot figure from per-fold metrics.

        :param result: Experiment result with multiple folds.
        """
        df = _build_df_from_experiment(result).sort_index()
        df["box"] = df["algorithm"] + "_" + df["rand_setting"] + "_" + df["test_mode"]
        df = df.dropna(axis=1, how="all")

        metrics = [m for m in _ALL_METRICS if "normalized" not in m and m in df.columns]
        if not metrics:
            logger.warning("violin: no metric has a finite value in any fold; the section will be skipped")

        self._fig = go.Figure()
        for metric in metrics:
            for box in df["box"].unique():
                tmp_df = df[df["box"] == box]
                label = box.split("_")[0] + ": " + metric
                self._fig.add_trace(
                    go.Violin(
                        y=tmp_df[metric],
                        x=[label] * len(tmp_df[metric]),
                        name=label,
                        box_visible=True,
                        meanline_visible=True,
                    )
                )

        self._fig.update_layout(title_text="All Metrics", height=600, width=1100)

        self._data = {}
        for model in result.models:
            for run in model.runs:
                sample_name = f"{model.model_name}_fold{run.fold_index}"
                self._data[sample_name] = dict(run.metrics)

    def to_png(self, path: str | Path) -> None:
        """Render violin plot to a static PNG.

        :param path: Output file path.
        """
        if self._fig is None:
            raise RuntimeError("Call compute() before to_png()")
        self._fig.write_image(str(path))

    def to_multiqc(self) -> list[Section]:
        """Return a MultiQC violin Section using native violin plot API."""
        if self._data is None:
            raise RuntimeError("Call compute() before to_multiqc()")
        try:
            from multiqc.plots import violin as mqc_violin
        except ImportError as e:
            raise ImportError("multiqc is required for to_multiqc(). Install with: pip install drevalpy[report]") from e

        metric_names = sorted({m for metrics in self._data.values() for m in metrics if _is_finite(metrics[m])})
        if not metric_names:
            logger.warning("violin: no metric has a finite value in any fold; skipping the section")
            return []
        headers: dict[str, dict[str, str]] = {m: {"title": m, "description": f"Metric: {m}"} for m in metric_names}
        plot = mqc_violin.plot(self._data, headers, pconfig={"id": "dreval_violin"})

        return [
            Section(
                name="Metric Distributions",
                anchor="dreval_violin",
                description="Distribution of evaluation metrics across cross-validation folds.",
                plot=plot,
            )
        ]

    def show(self) -> None:
        """Display the violin plot in a Jupyter notebook."""
        if self._fig is None:
            raise RuntimeError("Call compute() before show()")
        self._fig.show()
