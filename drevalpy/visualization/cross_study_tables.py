"""Module for generating evaluation tables for cross-study drug response prediction."""

import os
import pathlib
from io import TextIOWrapper

import pandas as pd
import plotly.graph_objects as go


class CrossStudyTables:
    """Generate evaluation tables for cross-study drug response prediction."""

    def __init__(self, evaluation_metrics: pd.DataFrame, path_data: pathlib.Path):
        """
        Initialize the CrossStudyTables object.

        :param evaluation_metrics: eval metrics dataframe.
        :param path_data: Path to data directory (used for context or extensions).
        """
        self.evaluation_metrics = evaluation_metrics
        self.path_data = path_data

        self.figures: dict[str, go.Figure] = {}
        cross_study_settings = evaluation_metrics[
            evaluation_metrics.rand_setting.str.contains("cross-study-")
        ].rand_setting.unique()
        self.cross_study_datasets = [setting.split("cross-study-")[1] for setting in cross_study_settings]

        evaluation_metrics = evaluation_metrics[evaluation_metrics.rand_setting.isin(cross_study_settings)]

        self.mean_metrics = []
        self.std_metrics = []
        for dataset in self.cross_study_datasets:
            evaluation_metrics_dataset = evaluation_metrics[
                evaluation_metrics.rand_setting.str.contains(f"cross-study-{dataset}")
            ]
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
            # sort by lowest MSE
            mean = mean.sort_values(by="MSE")
            std = std.loc[mean.index]

            mean.index = [s.split("_cross-study")[0] for s in mean.index]
            std.index = mean.index
            self.mean_metrics.append(mean)
            self.std_metrics.append(std)

    def draw(self):
        """Create and store Plotly table figures sorted by MSE."""
        for dataset_name, mean_df, std_df in zip(self.cross_study_datasets, self.mean_metrics, self.std_metrics):

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

    def draw_and_save(self, out_prefix: str, out_suffix: str):
        """
        Generate and save HTML tables for each cross-study dataset.

        :param out_prefix: Directory to save output files.
        :param out_suffix: Suffix to append to each output filename.
        """
        os.makedirs(out_prefix, exist_ok=True)
        self.draw()
        for dataset_name, fig in self.figures.items():
            filename = f"{out_prefix}/table_cross_study_{dataset_name}_{out_suffix}.html"
            fig.write_html(filename, include_plotlyjs="embed", full_html=True)

    @staticmethod
    def write_to_html(test_mode: str, f: TextIOWrapper, files: list[str], prefix: str) -> TextIOWrapper:
        """
        Embed HTML table files into an open HTML file handle.

        :param test_mode: Substring to match filenames (e.g., 'lpo', 'lco').
        :param f: Open writable file handle to insert HTML blocks.
        :param files: List of filenames in the target directory.
        :param prefix: Path prefix to locate HTML table files.

        :return: Updated file handle with HTML blocks written in.
        """
        if prefix:
            prefix = os.path.join(prefix, "html_tables")
        os.makedirs(prefix, exist_ok=True)

        for file in files:
            if file.startswith("table_cross_study_") and file.endswith(".html") and test_mode in file:
                f.write(f'<iframe src="html_tables/{file}" width="100%" height="600" frameborder="0"></iframe>\n')
        return f
