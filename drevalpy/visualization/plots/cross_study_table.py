"""Cross-study table visualization (Plotly table + MultiQC table)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.registry.visualization import register
from drevalpy.visualization.base import PlotlyVisualization, Section
from drevalpy.visualization.plots._utils import runs_frame

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go

    from drevalpy.types.results import ExperimentResult

_METRICS = [
    "MSE",
    "RMSE",
    "MAE",
    "R^2",
    "Pearson",
    "Spearman",
    "Kendall",
    "Pearson: normalized",
    "Spearman: normalized",
    "Kendall: normalized",
    "R^2: normalized",
]


@register(
    "cross_study_table",
    "Summary table of model metrics for cross-study evaluation",
)
class CrossStudyTableVisualization(PlotlyVisualization):
    """Tabular summary of model performance for cross-study predictions (Plotly table)."""

    def __init__(self) -> None:
        """Initialize with empty state."""
        self._fig: go.Figure | None = None
        self._result: ExperimentResult | None = None
        self._figures: dict[str, go.Figure] = {}
        self._mean_metrics: list[pd.DataFrame] = []
        self._std_metrics: list[pd.DataFrame] = []
        self._cross_study_datasets: list[str] = []

    def compute(self, result: ExperimentResult, dataset=None) -> None:
        """Build data for cross-study evaluation tables.

        :param result: Experiment result to summarize.
        """
        import plotly.graph_objects as go

        self._result = result
        df = runs_frame(result, indexed=True)

        cross_study_settings = df[df["rand_setting"].str.contains("cross-study-")]["rand_setting"].unique()
        self._cross_study_datasets = [s.split("cross-study-")[1] for s in cross_study_settings]

        filtered = df[df["rand_setting"].isin(cross_study_settings)]

        self._mean_metrics = []
        self._std_metrics = []
        for dataset in self._cross_study_datasets:
            ds_df = filtered[filtered["rand_setting"].str.contains(f"cross-study-{dataset}")]
            groups = [s.split("_split_")[0] for s in ds_df.index]
            available_metrics = [m for m in _METRICS if m in ds_df.columns]
            grouped = ds_df[available_metrics].groupby(groups)
            mean = grouped.mean()
            std = grouped.std()
            if "MSE" in mean.columns:
                mean = mean.sort_values(by="MSE")
                std = std.loc[mean.index]
            mean.index = [s.split("_cross-study")[0] for s in mean.index]
            std.index = mean.index
            self._mean_metrics.append(mean)
            self._std_metrics.append(std)

        self._figures = {}
        for dataset_name, mean_df, std_df in zip(
            self._cross_study_datasets, self._mean_metrics, self._std_metrics, strict=False
        ):
            formatted = mean_df.map(lambda x: f"{x:.3f}") + " ± " + std_df.map(lambda x: f"{x:.3f}")
            fig = go.Figure(
                data=[
                    go.Table(
                        header={
                            "values": ["Model"] + list(formatted.columns),
                            "fill_color": "lightgrey",
                            "align": "left",
                        },
                        cells={
                            "values": [formatted.index] + [formatted[col].values for col in formatted.columns],
                            "fill_color": "white",
                            "align": "left",
                        },
                    )
                ]
            )
            fig.update_layout(title_text=f"Evaluation Metrics for Cross-Study Predictions to {dataset_name}")
            self._figures[dataset_name] = fig

        if self._figures:
            self._fig = next(iter(self._figures.values()))
        else:
            self._fig = _build_simple_table(result)

    def to_multiqc(self) -> list[Section]:
        """Return MultiQC table Sections."""
        if self._result is None:
            raise RuntimeError("Call compute() before to_multiqc()")
        try:
            from multiqc.plots import table as mqc_table
        except ImportError as e:
            raise ImportError("multiqc is required for to_multiqc(). Install with: pip install drevalpy[report]") from e

        metric_names = sorted({m for model in self._result.models for m in model.aggregate_metrics})

        table_data: dict[str, dict[str, float]] = {}
        for model in self._result.models:
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


def _build_simple_table(result: ExperimentResult) -> go.Figure:
    """Build a simple Plotly table from aggregate metrics when no cross-study data is present."""
    import plotly.graph_objects as go

    metric_names = sorted({m for model in result.models for m in model.aggregate_metrics})
    model_names = [m.model_name for m in result.models]

    header_vals = ["Model"] + metric_names
    cell_values: list[list] = [model_names]
    for metric in metric_names:
        col: list[str] = []
        for model in result.models:
            agg = model.aggregate_metrics.get(metric)
            if agg:
                col.append(f"{agg['mean']:.3f} ± {agg['std']:.3f}")
            else:
                col.append("N/A")
        cell_values.append(col)

    fig = go.Figure(
        data=[
            go.Table(
                header={"values": header_vals, "fill_color": "lightgrey", "align": "left"},
                cells={"values": cell_values, "fill_color": "white", "align": "left"},
            )
        ]
    )
    fig.update_layout(title_text="Model Performance Summary")
    return fig
