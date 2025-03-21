"""Module for generating evaluation tables for cross-study drug response prediction."""

import os
import pathlib

import pandas as pd
import plotly.graph_objects as go

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.evaluation import evaluate

from ..pipeline_function import pipeline_function


class CrossStudyTables:
    """Generate evaluation tables for cross-study drug response prediction."""

    def __init__(self, true_vs_pred: pd.DataFrame, path_data: pathlib.Path):
        """
        Initialize the CrossStudyTables object.

        :param true_vs_pred: DataFrame with [drug, cell_line, y_true, y_pred, algorithm, rand_setting, CV_split].
        :param path_data: Path to data directory (used for context or extensions).
        """
        self.true_vs_pred = true_vs_pred
        self.path_data = path_data
        self.cross_study_settings = [
            setting for setting in true_vs_pred.rand_setting.unique() if "cross-study-" in setting
        ]
        self.cross_study_datasets = [setting.split("cross-study-")[1] for setting in self.cross_study_settings]
        self.models = true_vs_pred.algorithm.unique()
        self.mean_resulting_dataframes = []
        self.std_resulting_dataframes = []
        self.figures = {}

    def compute_metrics(self):
        """Compute mean and standard deviation of evaluation metrics across CV splits."""
        for dataset in self.cross_study_datasets:
            results = {}
            cs_data = self.true_vs_pred[self.true_vs_pred.rand_setting == f"cross-study-{dataset}"]

            for model in self.models:
                model_results = []
                for split in cs_data.CV_split.unique():
                    cs_data_model = cs_data[(cs_data.algorithm == model) & (cs_data.CV_split == split)]
                    ds = DrugResponseDataset(
                        drug_ids=cs_data_model.drug,
                        cell_line_ids=cs_data_model.cell_line,
                        response=cs_data_model.y_true.values,
                        predictions=cs_data_model.y_pred.values,
                    )
                    evaluation_metrics_dicts = evaluate(
                        ds, metric=["MSE", "RMSE", "MAE", "R^2", "Pearson", "Spearman", "Kendall"]
                    )
                    model_results.append(evaluation_metrics_dicts)

                df_model = pd.DataFrame(model_results)
                results[model] = {"mean": df_model.mean(), "std": df_model.std()}

            mean_df = pd.DataFrame({model: results[model]["mean"] for model in results}).T
            std_df = pd.DataFrame({model: results[model]["std"] for model in results}).T

            self.mean_resulting_dataframes.append(mean_df)
            self.std_resulting_dataframes.append(std_df)

    def draw(self):
        """Create and store Plotly table figures sorted by MSE."""
        for dataset_name, mean_df, std_df in zip(
            self.cross_study_datasets, self.mean_resulting_dataframes, self.std_resulting_dataframes
        ):
            mean_df_sorted = mean_df.sort_values(by="MSE")
            std_df_sorted = std_df.loc[mean_df_sorted.index]
            formatted_data = mean_df_sorted.map(lambda x: f"{x:.3f}") + " ± " + std_df_sorted.map(lambda x: f"{x:.3f}")

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
            fig.update_layout(title_text=f"Evaluation Metrics for Cross-Study {dataset_name}")
            self.figures[dataset_name] = fig

    @pipeline_function
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
    def write_to_html(lpo_lco_ldo: str, f, files: list[str], prefix: str):
        """
        Embed HTML table files into an open HTML file handle.

        :param lpo_lco_ldo: Substring to match filenames (e.g., 'lpo', 'lco').
        :param f: Open writable file handle to insert HTML blocks.
        :param files: List of filenames in the target directory.
        :param prefix: Path prefix to locate HTML table files.

        :return: Updated file handle with HTML blocks written in.
        """
        if prefix:
            prefix = os.path.join(prefix, "html_tables")
        os.makedirs(prefix, exist_ok=True)

        for file in files:
            if file.startswith("table_cross_study_") and file.endswith(".html") and lpo_lco_ldo in file:
                f.write(f'<h2 id="tables">Evaluation Results: {file}</h2>\n')
                f.write(f'<iframe src="html_tables/{file}" width="100%" height="600" frameborder="0"></iframe>\n')
        return f
