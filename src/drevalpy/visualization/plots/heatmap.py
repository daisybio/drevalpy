"""Heatmap visualization (Plotly + MultiQC heatmap)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drevalpy.evaluation import Metric, MetricKind
from drevalpy.log import get_logger
from drevalpy.registry.visualization import register
from drevalpy.visualization.base import Plot, Section
from drevalpy.visualization.plots._utils import runs_frame
from drevalpy.visualization.requirements import PlotRequirement

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go
    from numpy.typing import ArrayLike

    from drevalpy.types.results import ExperimentResult

logger = get_logger(__name__)


def compute_ssmd(values_a: ArrayLike, values_b: ArrayLike) -> float:
    """Compute the Strictly Standardized Mean Difference between two arrays.

    SSMD = (mean_a - mean_b) / sqrt(var_a + var_b)

    :param values_a: Metric values for model A (across CV folds).
    :param values_b: Metric values for model B (across CV folds).
    :returns: SSMD value, or NaN if either array has fewer than two values or the denominator is zero.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    denom = var_a + var_b
    if denom <= 0:
        return float("nan")
    return float((mu_a - mu_b) / np.sqrt(denom))


def _ssmd_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Build the pairwise SSMD matrix of the models on *metric*.

    :param df: Per-run frame from :func:`~drevalpy.visualization.plots._utils.runs_frame`.
    :param metric: Metric column to compare the models on.
    :returns: Models by models SSMD matrix, or an empty frame if *metric* is not a column.
    """
    import pandas as pd

    if metric not in df.columns:
        return pd.DataFrame()

    values = {model: group.astype(float).dropna() for model, group in df.groupby("algorithm", sort=False)[metric]}
    matrix = pd.DataFrame(index=list(values), columns=list(values), dtype=float)
    for m1, values_m1 in values.items():
        for m2, values_m2 in values.items():
            matrix.loc[m1, m2] = 0.0 if m1 == m2 else compute_ssmd(values_m1, values_m2)
    return matrix.astype(float)


def _ssmd_trace(df: pd.DataFrame, metric_name: str) -> go.Heatmap | None:
    import plotly.graph_objects as go

    dt = _ssmd_matrix(df, metric_name)
    if dt.empty:
        return None
    dt["sort_key"] = dt.max(axis=1)
    dt = dt.sort_values(by="sort_key", ascending=True).drop(columns=["sort_key"])
    dt = dt[dt.index]
    text_labels = dt.round(3).astype(str)
    return go.Heatmap(
        z=dt.values,
        x=list(dt.columns),
        y=list(dt.index),
        colorscale="RdBu",
        texttemplate="%{text}",
        text=text_labels.values,
        textfont={"size": 12},
        showscale=False,
    )


def _mean_trace(df: pd.DataFrame, ps: str, metric_cols: list[str]) -> go.Heatmap | None:
    import plotly.graph_objects as go

    columns = _columns_for_setting(ps, metric_cols)
    if not columns:
        return None
    colorscale = {"r2": "Blues", "correlations": "Viridis", "errors": "hot"}[ps]
    ascending = ps != "errors"

    grouped = df.astype(dict.fromkeys(columns, float)).groupby(["algorithm", "rand_setting", "test_mode"])[columns]
    dt = grouped.mean().sort_values(by=columns[0], ascending=ascending)
    dt_std = grouped.sem().fillna(0.0).loc[dt.index]
    text_labels = dt.round(3).astype(str) + " ± " + dt_std.round(3).astype(str)
    return go.Heatmap(
        z=dt.values,
        x=list(dt.columns),
        y=list(dt.index.get_level_values("algorithm")),
        colorscale=colorscale,
        texttemplate="%{text}",
        text=text_labels.values,
        textfont={"size": 12},
        showscale=False,
    )


@register("heatmap", "Heatmap of mean metrics per model")
class HeatmapPlot(Plot):
    """Heatmap showing mean metric values (rows=models, cols=metrics) with SSMD subplots."""

    requirements = frozenset({PlotRequirement.MULTIPLE_FOLDS})

    def figure(self, result: ExperimentResult) -> go.Figure:
        """Draw the Plotly heatmap figure with mean metrics and SSMD panels.

        :param result: Experiment result with multiple folds.
        :returns: The Plotly figure.
        """
        from plotly.subplots import make_subplots

        df = runs_frame(result)
        metric_cols = [m.value for m in Metric if m in df.columns]
        if not metric_cols:
            logger.warning("heatmap: none of the expected metrics are present; the panels will be empty")

        plot_settings = ["r2", "correlations", "errors", "ssmd_R^2", "ssmd_MSE"]
        titles = [
            "Mean R^2",
            "Mean Correlations",
            "Mean Errors",
            "SSMD for R^2",
            "SSMD for MSE",
        ]

        fig = make_subplots(
            rows=len(plot_settings),
            cols=1,
            subplot_titles=tuple(titles),
            vertical_spacing=0.1,
        )

        for idx, ps in enumerate(plot_settings, start=1):
            if ps.startswith("ssmd_"):
                trace = _ssmd_trace(df, ps.split("_", 1)[1])
            else:
                trace = _mean_trace(df, ps, metric_cols)
            if trace is not None:
                fig.add_trace(trace, row=idx, col=1)

        n_models = len(result.models)
        height_per_model = 35
        new_height = min(500 + n_models * height_per_model, 5000)
        fig.update_layout(
            height=new_height,
            width=1300,
            title_text="Heatmap of the evaluation metrics",
        )
        return fig

    def sections(self, result: ExperimentResult) -> list[Section]:
        """Return a MultiQC heatmap Section using the native heatmap plot API.

        :param result: Experiment result with multiple folds.
        :returns: A single-element list.
        """
        from multiqc.plots import heatmap as mqc_heatmap

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
            pconfig={"id": "dreval_heatmap", "title": "Model Performance Heatmap", "square": False},
        )

        return [
            Section(
                name="Performance Heatmap",
                anchor="dreval_heatmap",
                description="Mean metric values per model across cross-validation folds.",
                plot=plot,
            )
        ]


_SETTING_KINDS = {"r2": MetricKind.FIT, "correlations": MetricKind.CORRELATION, "errors": MetricKind.ERROR}


def _columns_for_setting(ps: str, metric_cols: list[str]) -> list[str]:
    kind = _SETTING_KINDS.get(ps)
    return [column for column in metric_cols if Metric(column).kind is kind]
