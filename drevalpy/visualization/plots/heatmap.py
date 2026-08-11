"""Heatmap visualization (Plotly + MultiQC heatmap)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from drevalpy.registry.visualization import register
from drevalpy.visualization.base import Section, Visualization
from drevalpy.visualization.requirements import PlotRequirement

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult

_ALL_METRICS = [
    "R^2",
    "Pearson",
    "Spearman",
    "Kendall",
    "MSE",
    "RMSE",
    "MAE",
]


def _build_df_from_experiment(result: ExperimentResult) -> pd.DataFrame:
    """Build a flat DataFrame from an ExperimentResult with an index encoding model/setting/split."""
    rows: list[dict] = []
    indices: list[str] = []
    for model in result.models:
        for run in model.runs:
            rand = f"{run.randomization[0]}_{run.randomization[1]}" if run.randomization else "predictions"
            row: dict = {
                "algorithm": run.model_name,
                "rand_setting": rand,
                "test_mode": result.split_mode,
                "CV_split": run.fold_index,
            }
            row.update(run.metrics)
            rows.append(row)
            indices.append(f"{run.model_name}_{rand}_{result.split_mode}_split_{run.fold_index}")
    df = pd.DataFrame(rows, index=indices)
    return df


def _setting_groups(df: pd.DataFrame) -> pd.Series:
    idx_split = df.index.to_series().str.split("_")
    return idx_split.str[0:3].str.join("_")


def _calc_summary_metric(x: pd.DataFrame, std_error: bool = False) -> pd.Series:
    results = pd.Series(index=x.columns, dtype=float)
    for col in x.columns:
        if np.count_nonzero(np.isnan(x[col].values.astype(float))) == len(x[col]):
            results[col] = np.nan
        elif std_error:
            results[col] = np.nanstd(x[col].values.astype(float)) / np.sqrt(x.shape[0])
        else:
            results[col] = np.nanmean(x[col].values.astype(float))
    return results


def _compute_ssmd(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["model_name"] = df.index.to_series().apply(lambda x: x.split("_predictions")[0])
    models = df["model_name"].unique()
    ssmd_matrix = pd.DataFrame(index=models, columns=models, dtype=float)

    for m1 in models:
        for m2 in models:
            if m1 == m2:
                ssmd_matrix.loc[m1, m2] = 0.0
                continue
            values_m1 = df[df["model_name"] == m1][metric].astype(float)
            values_m2 = df[df["model_name"] == m2][metric].astype(float)
            mu1, mu2 = values_m1.mean(), values_m2.mean()
            sigma1_sq, sigma2_sq = values_m1.var(ddof=1), values_m2.var(ddof=1)
            denom = sigma1_sq + sigma2_sq
            ssmd_matrix.loc[m1, m2] = (mu1 - mu2) / np.sqrt(denom) if denom > 0 else np.nan

    return ssmd_matrix.astype(float)


@register(
    "heatmap",
    "Heatmap of mean metrics per model",
    requirements=frozenset({PlotRequirement.MULTIPLE_FOLDS}),
)
class HeatmapVisualization(Visualization):
    """Heatmap showing mean metric values (rows=models, cols=metrics) with SSMD subplots."""

    def __init__(self) -> None:
        """Initialize with empty state."""
        self._fig: go.Figure | None = None
        self._result: ExperimentResult | None = None

    def compute(self, result: ExperimentResult, dataset=None) -> None:
        """Build the Plotly heatmap figure with mean metrics and SSMD panels.

        :param result: Experiment result with multiple folds.
        """
        self._result = result
        df = _build_df_from_experiment(result)
        metric_cols = [m for m in _ALL_METRICS if m in df.columns]
        df_metrics = df[metric_cols]

        setting = _setting_groups(df)

        plot_settings = ["r2", "correlations", "errors", "ssmd_R^2", "ssmd_MSE"]
        titles = [
            "Mean R^2",
            "Mean Correlations",
            "Mean Errors",
            "SSMD for R^2",
            "SSMD for MSE",
        ]

        self._fig = make_subplots(
            rows=len(plot_settings),
            cols=1,
            subplot_titles=tuple(titles),
            vertical_spacing=0.1,
        )

        for idx, ps in enumerate(plot_settings, start=1):
            if ps.startswith("ssmd_"):
                metric_name = ps.split("_", 1)[1]
                dt = _compute_ssmd(df, metric_name)
                if dt.empty:
                    continue
                dt["sort_key"] = dt.max(axis=1)
                dt = dt.sort_values(by="sort_key", ascending=True).drop(columns=["sort_key"])
                dt = dt[dt.index]
                text_labels = dt.round(3).astype(str)
                labels = list(dt.index)
                self._fig.add_trace(
                    go.Heatmap(
                        z=dt.values,
                        x=list(dt.columns),
                        y=labels,
                        colorscale="RdBu",
                        texttemplate="%{text}",
                        text=text_labels.values,
                        textfont={"size": 12},
                        showscale=False,
                    ),
                    row=idx,
                    col=1,
                )
            else:
                columns = _columns_for_setting(ps, metric_cols)
                if not columns:
                    continue
                colorscale = {"r2": "Blues", "correlations": "Viridis", "errors": "hot"}[ps]
                ascending = ps != "errors"

                dt = df_metrics[columns].groupby(setting).apply(lambda x: _calc_summary_metric(x))
                dt = dt.sort_values(by=columns[0], ascending=ascending)
                dt_std = df_metrics[columns].groupby(setting).apply(lambda x: _calc_summary_metric(x, std_error=True))
                dt_std = dt_std.loc[dt.index]
                text_labels = dt.round(3).astype(str) + " ± " + dt_std.round(3).astype(str)
                labels = [i.split("_")[0] for i in dt.index]

                self._fig.add_trace(
                    go.Heatmap(
                        z=dt.values,
                        x=list(dt.columns),
                        y=labels,
                        colorscale=colorscale,
                        texttemplate="%{text}",
                        text=text_labels.values,
                        textfont={"size": 12},
                        showscale=False,
                    ),
                    row=idx,
                    col=1,
                )

        n_models = len(result.models)
        height_per_model = 35
        new_height = min(500 + n_models * height_per_model, 5000)
        self._fig.update_layout(
            height=new_height,
            width=1300,
            title_text="Heatmap of the evaluation metrics",
        )

    def to_png(self, path: str | Path) -> None:
        """Render heatmap to a static PNG.

        :param path: Output file path.
        """
        if self._fig is None:
            raise RuntimeError("Call compute() before to_png()")
        self._fig.write_image(str(path))

    def to_multiqc(self) -> list[Section]:
        """Return a MultiQC heatmap Section using native heatmap plot API."""
        if self._result is None:
            raise RuntimeError("Call compute() before to_multiqc()")
        try:
            from multiqc.plots import heatmap as mqc_heatmap
        except ImportError as e:
            raise ImportError("multiqc is required for to_multiqc(). Install with: pip install drevalpy[report]") from e

        metric_names = sorted({m for model in self._result.models for m in model.aggregate_metrics})
        model_names = [m.model_name for m in self._result.models]

        data: list[list[float | None]] = []
        for model in self._result.models:
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

    def show(self) -> None:
        """Display the heatmap in a Jupyter notebook."""
        if self._fig is None:
            raise RuntimeError("Call compute() before show()")
        self._fig.show()


def _columns_for_setting(ps: str, metric_cols: list[str]) -> list[str]:
    if ps == "r2":
        return [c for c in metric_cols if "R^2" in c]
    if ps == "correlations":
        return [c for c in metric_cols if "Pearson" in c or "Spearman" in c or "Kendall" in c]
    if ps == "errors":
        return [c for c in metric_cols if c in ("MSE", "RMSE", "MAE")]
    return []
