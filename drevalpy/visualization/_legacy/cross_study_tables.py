"""Module for generating evaluation tables for cross-study drug response prediction."""

from __future__ import annotations

import pathlib
from io import TextIOWrapper
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
from upath import UPath as Path

from .outplot import OutPlot

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult


def _build_cross_study_df_from_experiment(result: ExperimentResult) -> pd.DataFrame:
    """Build a DataFrame from an ExperimentResult for cross-study tables.

    Columns: algorithm, rand_setting, test_mode, CV_split, and metric columns.
    The index encodes model/setting/split for grouping.
    """
    rows: list[dict] = []
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
            idx = f"{run.model_name}_{rand}_{result.split_mode}_split_{run.fold_index}"
            row["_index"] = idx
            rows.append(row)
    df = pd.DataFrame(rows)
    if "_index" in df.columns:
        df = df.set_index("_index")
        df.index.name = None
    return df


class CrossStudyTables(OutPlot):
    """Generate evaluation tables for cross-study drug response prediction."""

    result_type: str = "ExperimentResult"
    requirements: frozenset = frozenset()

    def __init__(
        self,
        result: ExperimentResult | None = None,
        *,
        evaluation_metrics: pd.DataFrame | None = None,
        path_data: pathlib.Path | None = None,
    ):
        """Initialize cross-study evaluation tables.

        :param result: Typed experiment result (preferred path).
        :param evaluation_metrics: Legacy aggregated evaluation metrics dataframe.
        :param path_data: Dataset root directory (reserved for extensions).
        """
        if result is not None:
            self.evaluation_metrics = _build_cross_study_df_from_experiment(result)
            self.path_data = None
        elif evaluation_metrics is not None:
            self.evaluation_metrics = evaluation_metrics
            self.path_data = path_data
        else:
            raise ValueError("Either 'result' or 'evaluation_metrics' must be provided")

        self.figures: dict[str, go.Figure] = {}
        cross_study_settings = self.evaluation_metrics[
            self.evaluation_metrics.rand_setting.str.contains("cross-study-")
        ].rand_setting.unique()
        self.cross_study_datasets = [setting.split("cross-study-")[1] for setting in cross_study_settings]

        filtered = self.evaluation_metrics[self.evaluation_metrics.rand_setting.isin(cross_study_settings)]

        self.mean_metrics = []
        self.std_metrics = []
        for dataset in self.cross_study_datasets:
            evaluation_metrics_dataset = filtered[filtered.rand_setting.str.contains(f"cross-study-{dataset}")]
            evaluation_metrics_group = [s.split("_split_")[0] for s in evaluation_metrics_dataset.index]
            metrics = [
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
            grouped = evaluation_metrics_dataset[metrics].groupby(evaluation_metrics_group)
            mean = grouped.mean()
            std = grouped.std()
            mean = mean.sort_values(by="MSE")
            std = std.loc[mean.index]

            mean.index = [s.split("_cross-study")[0] for s in mean.index]
            std.index = mean.index
            self.mean_metrics.append(mean)
            self.std_metrics.append(std)

    def _draw(self) -> None:
        """Create and store Plotly table figures sorted by MSE."""
        self.draw()

    def draw(self):
        """Create and store Plotly table figures sorted by MSE."""
        for dataset_name, mean_df, std_df in zip(
            self.cross_study_datasets, self.mean_metrics, self.std_metrics, strict=False
        ):
            formatted_data = mean_df.map(lambda x: f"{x:.3f}") + " ± " + std_df.map(lambda x: f"{x:.3f}")

            fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=["Model"] + list(formatted_data.columns), fill_color="lightgrey", align="left"
                        ),
                        cells=dict(
                            values=[formatted_data.index]
                            + [formatted_data[col].values for col in formatted_data.columns],
                            fill_color="white",
                            align="left",
                        ),
                    )
                ]
            )
            fig.update_layout(title_text=f"Evaluation Metrics for Cross-Study Predictions to {dataset_name}")
            self.figures[dataset_name] = fig

    def draw_and_save(self, out_prefix: str | Path, out_suffix: str):
        """Generate and save HTML tables for each cross-study dataset.

        :param out_prefix: Directory for output HTML files.
        :param out_suffix: Suffix appended to each output filename.
        """
        out_dir = Path(out_prefix)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.draw()
        for dataset_name, fig in self.figures.items():
            filename = out_dir / f"table_cross_study_{dataset_name}_{out_suffix}.html"
            fig.write_html(filename, include_plotlyjs="embed", full_html=True)

    @staticmethod
    def write_to_html(test_mode: str, f: TextIOWrapper, files: list[str], prefix: str | Path) -> TextIOWrapper:
        """Embed cross-study table iframes into the report HTML.

        :param test_mode: Substring to match filenames (for example ``"LCO"``).
        :param f: Open writable HTML file handle.
        :param files: Filenames in the html_tables directory.
        :param prefix: Path prefix to locate table files.

        :returns: The same file handle after writing.
        """
        table_dir = Path(prefix) / "html_tables" if prefix else Path()
        table_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            if file.startswith("table_cross_study_") and file.endswith(".html") and test_mode in file:
                f.write(f'<iframe src="html_tables/{file}" width="100%" height="600" frameborder="0"></iframe>\n')
        return f
