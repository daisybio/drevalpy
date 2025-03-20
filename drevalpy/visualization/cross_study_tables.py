import json
import os
import pathlib

import pandas as pd
import plotly.graph_objects as go

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.evaluation import evaluate


class CrossStudyTables:
    """Generates evaluation tables for cross-study drug response prediction."""

    def __init__(self, true_vs_pred: pd.DataFrame, path_data: pathlib.Path):
        """
        Initialize the evaluation class.

        :param true_vs_pred: DataFrame containing true vs. predicted values.
        :param path_data: Path to the data directory.
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

    def compute_metrics(self):
        """Compute evaluation metrics (mean and std over splits)."""
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

    def plot_results(self):
        """Generate Plotly tables displaying evaluation metrics sorted by MSE."""
        plotly_tables = []

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
            fig.update_layout(title_text=f"Evaluation Metrics for Cross-Study {dataset_name} (Sorted by MSE)")
            plotly_tables.append(fig)

        for fig in plotly_tables:
            fig.show()

    def save_to_json(self, out_prefix: str, out_suffix: str):
        """Save the evaluation metrics as JSON files."""
        for dataset_name, mean_df, std_df in zip(
            self.cross_study_datasets, self.mean_resulting_dataframes, self.std_resulting_dataframes
        ):
            path_out = f"{out_prefix}table_{dataset_name}_{out_suffix}.json"
            output_json = {
                "data": mean_df.fillna("").to_dict(orient="records"),
                "std": std_df.fillna("").to_dict(orient="records"),
            }
            with open(path_out, "w") as f:
                json.dump(output_json, f, indent=4)

    @staticmethod
    def write_to_html(f, prefix: str, files: list[str], dataset_name: str):
        """Write evaluation results into an HTML file."""
        if prefix:
            prefix = os.path.join(prefix, "html_tables")
        f.write(f'<h2 id="tables"> Evaluation Results for {dataset_name}</h2>\n')
        table_file = f"table_{dataset_name}.json"
        if table_file in files:
            json_file = pd.read_json(pathlib.Path(prefix, table_file))
            f.write('<table id="summaryTable" class="display" style="width:100%">\n')
            f.write("<thead>\n<tr>\n")
            for col in json_file.columns:
                f.write(f"<th>{col}</th>\n")
            f.write("</tr>\n</thead>\n<tbody></tbody>\n</table>\n")
