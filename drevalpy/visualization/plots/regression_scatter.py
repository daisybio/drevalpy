"""Regression scatter plot visualization (Plotly + MultiQC scatter)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr

from drevalpy.visualization.base import Section, Visualization
from drevalpy.visualization.registry import visualization_registry

if TYPE_CHECKING:
    from drevalpy.types.results import ModelResult


def _build_regression_df(result: ModelResult) -> pd.DataFrame:
    """Build DataFrame with y_true, y_pred, fold, drug, cell line columns."""
    rows: list[dict] = []
    for run in result.runs:
        if run.randomization is not None:
            continue
        for i in range(len(run.predictions)):
            rows.append(
                {
                    "y_true": float(run.ground_truth[i]),
                    "y_pred": float(run.predictions[i]),
                    "algorithm": run.model_name,
                    "CV_split": run.fold_index,
                    "drug_name": str(run.drug_ids[i]),
                    "cell_line_name": str(run.cell_line_ids[i]),
                }
            )
    return pd.DataFrame(rows)


@visualization_registry.register(
    "regression_scatter",
    "Scatter plot of predicted vs. actual drug response values",
    result_type="ModelResult",
)
class RegressionScatterVisualization(Visualization):
    """Predicted vs. ground-truth scatter for a single model (Plotly)."""

    def __init__(self) -> None:
        """Initialize with empty state."""
        self._fig: go.Figure | None = None
        self._result: ModelResult | None = None
        self._group_by: str = "drug_name"

    def compute(self, result: ModelResult, dataset=None, group_by: str = "drug_name") -> None:
        """Build Plotly scatter figure showing ground truth vs predictions per group.

        :param result: Model result containing predictions and ground truth.
        :param group_by: Column to group and color by ('drug_name' or 'cell_line_name').
        """
        self._result = result
        self._group_by = group_by

        df = _build_regression_df(result)
        if df.empty:
            self._fig = go.Figure()
            return

        df = df.groupby(group_by).filter(lambda x: len(x) > 1)
        pccs = df.groupby(group_by).apply(lambda x: pearsonr(x["y_true"], x["y_pred"])[0], include_groups=False)
        pccs = pccs.reset_index()
        pccs.columns = [group_by, "pcc"]
        df = df.merge(pccs, on=group_by)

        df = df.sort_values(group_by)
        setting_title = result.model_name
        hover_data = ["pcc", "cell_line_name", "drug_name", "algorithm"]

        self._fig = px.scatter(
            df,
            x="y_true",
            y="y_pred",
            color=group_by,
            trendline="ols",
            hover_name=group_by,
            hover_data=hover_data,
            title=f"{setting_title}: Regression plot",
        )

        min_val = min(df["y_true"].min(), df["y_pred"].min())
        max_val = max(df["y_true"].max(), df["y_pred"].max())
        self._fig.update_xaxes(range=[min_val, max_val])
        self._fig.update_yaxes(range=[min_val, max_val])

    def to_png(self, path: str | Path) -> None:
        """Render scatter plot to a static PNG.

        :param path: Output file path.
        """
        if self._fig is None:
            raise RuntimeError("Call compute() before to_png()")
        self._fig.write_image(str(path))

    def to_multiqc(self) -> list[Section]:
        """Return a MultiQC scatter Section using native scatter plot API."""
        if self._result is None:
            raise RuntimeError("Call compute() before to_multiqc()")
        try:
            from multiqc.plots import scatter as mqc_scatter
        except ImportError as e:
            raise ImportError("multiqc is required for to_multiqc(). Install with: pip install drevalpy[report]") from e

        datasets: list[dict[str, list[dict[str, float]]]] = []
        for run in self._result.runs:
            if run.randomization is not None:
                continue
            mask = ~np.isnan(run.ground_truth) & ~np.isnan(run.predictions)
            gt = run.ground_truth[mask]
            pred = run.predictions[mask]
            points = [{"x": float(g), "y": float(p)} for g, p in zip(gt, pred, strict=True)]
            datasets.append({f"fold_{run.fold_index}": points})

        plot = mqc_scatter.plot(
            datasets,
            pconfig={
                "id": f"dreval_scatter_{self._result.model_name}",
                "title": f"Regression Scatter: {self._result.model_name}",
                "xlab": "Ground Truth",
                "ylab": "Predicted",
            },
        )

        return [
            Section(
                name=f"Regression Scatter: {self._result.model_name}",
                anchor=f"dreval_scatter_{self._result.model_name}",
                description=(
                    f"Predicted vs. ground-truth values for {self._result.model_name} "
                    f"across {self._result.n_folds} fold(s)."
                ),
                plot=plot,
            )
        ]

    def show(self) -> None:
        """Display the scatter plot in a Jupyter notebook."""
        if self._fig is None:
            raise RuntimeError("Call compute() before show()")
        self._fig.show()
