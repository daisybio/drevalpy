"""Renders the evaluation results as HTML tables."""

import json
import os
import pathlib
from io import TextIOWrapper

import pandas as pd

from ..pipeline_function import pipeline_function
from .outplot import OutPlot


class HTMLTable(OutPlot):
    """Renders the evaluation results as HTML tables."""

    @pipeline_function
    def __init__(self, df: pd.DataFrame, group_by: str, dataset: str, path_data: pathlib.Path) -> None:
        """
        Initialize the HTMLTable class.

        :param df: either all results of a setting or results evaluated by group (cell line, drug) for a setting
        :param group_by: all or the group by which the results are evaluated
        :param dataset: dataset name
        :param path_data: path to the data
        """
        self.df = df
        self.group_by = group_by
        self.dataset = dataset
        self.path_data = path_data

    @pipeline_function
    def draw_and_save(self, out_prefix: str, out_suffix: str) -> None:
        """
        Draw the table and save it to a file.

        :param out_prefix: e.g., results/my_run/html_tables/
        :param out_suffix: e.g., LPO, LPO_drug
        """
        self._draw()
        path_out = f"{out_prefix}table_{out_suffix}.json"
        # replace NaN with ""
        self.df = self.df.fillna("")
        output_json = {"data": self.df.to_dict(orient="records")}
        with open(path_out, "w") as f:
            json.dump(output_json, f, indent=4)

    def _draw(self) -> None:
        """Draw the table."""
        selected_columns = [
            "algorithm",
            "rand_setting",
            "CV_split",
            "MSE",
            "R^2",
            "Pearson",
            "RMSE",
            "MAE",
            "Spearman",
            "Kendall",
        ]
        if self.group_by == "drug":
            selected_columns = ["drug"] + selected_columns
        elif self.group_by == "cell_line":
            selected_columns = ["cell_line"] + selected_columns
        else:
            selected_columns = [
                "algorithm",
                "rand_setting",
                "CV_split",
                "MSE",
                "R^2",
                "Pearson",
                "R^2: normalized",
                "Pearson: normalized",
                "RMSE",
                "MAE",
                "Spearman",
                "Kendall",
                "Spearman: normalized",
                "Kendall: normalized",
            ]
            # only take the columns that occur
            selected_columns = [col for col in selected_columns if col in self.df.columns]
        # reorder columns
        self.df = self.df[selected_columns]
        # collapse measures over CV splits
        categorical_columns = [
            "algorithm",
            "rand_setting",
            "drug",
            "cell_line",
        ]
        categorical_columns = [col for col in categorical_columns if col in self.df.columns]
        self.df = self.df.groupby(categorical_columns).mean().reset_index()
        self.df = self.df.drop(columns=["CV_split"])
        if self.group_by == "drug":
            # link to pubchem
            # read in drug_names
            drug_names = pd.read_csv(pathlib.Path(f"{self.path_data}/{self.dataset}/drug_names.csv"))
            for cross_datasets in self.df["rand_setting"].unique():
                if "cross" in cross_datasets:
                    cs = cross_datasets.split("cross-study-")[1]
                    dn = pd.read_csv(pathlib.Path(f"{self.path_data}/{cs}/drug_names.csv"))
                    drug_names = pd.concat([drug_names, dn])
            drug_names = drug_names.drop_duplicates()
            self.df = self.df.merge(drug_names, left_on="drug", right_on="pubchem_id", how="left")
            self.df = self.df.drop(columns=["pubchem_id"])
        elif self.group_by == "cell_line":
            # link to cellosaurus
            cell_line_ids = pd.read_csv(pathlib.Path(f"{self.path_data}/{self.dataset}/cell_line_names.csv"))
            for cross_datasets in self.df["rand_setting"].unique():
                if "cross" in cross_datasets:
                    cs = cross_datasets.split("cross-study-")[1]
                    cl = pd.read_csv(pathlib.Path(f"{self.path_data}/{cs}/cell_line_names.csv"))
                    cell_line_ids = pd.concat([cell_line_ids, cl])
            cell_line_ids = cell_line_ids.drop_duplicates()
            self.df = self.df.merge(cell_line_ids, left_on="cell_line", right_on="cell_line_name", how="left")
            self.df = self.df.drop(columns=["cell_line_name"])

    @staticmethod
    def write_to_html(lpo_lco_ldo: str, f: TextIOWrapper, prefix: str = "", *args, **kwargs) -> TextIOWrapper:
        """
        Write the evaluation results into the report HTML file.

        :param lpo_lco_ldo: setting, e.g., LPO
        :param f: report file
        :param prefix: e.g., results/my_run
        :param args: additional arguments
        :param kwargs: additional keyword arguments
        :return: the report file
        """
        files: list[str] = kwargs.get("files", [])
        if prefix != "":
            prefix = os.path.join(prefix, "html_tables")
        f.write('<h2 id="tables"> Evaluation Results Table</h2>\n')
        whole_table = _get_table(files=files, file_table=f"table_{lpo_lco_ldo}.json")
        _write_table(f=f, table=whole_table, name="summaryTable", prefix=prefix)

        if lpo_lco_ldo != "LCO":
            f.write("<h2> Evaluation Results per Cell Line Table</h2>\n")
            cell_line_table = _get_table(files=files, file_table=f"table_cell_line_{lpo_lco_ldo}.json")
            _write_table(f=f, table=cell_line_table, name="clTable", prefix=prefix)
        if lpo_lco_ldo != "LDO":
            f.write("<h2> Evaluation Results per Drug Table</h2>\n")
            drug_table = _get_table(files=files, file_table=f"table_drug_{lpo_lco_ldo}.json")
            _write_table(f=f, table=drug_table, name="drugTable", prefix=prefix)
        return f


def _write_table(f: TextIOWrapper, table: str, name: str, prefix: str = ""):
    json_file = pd.read_json(pathlib.Path(prefix, table))
    f.write(f'<table id="{name}" class="display" style="width:100%">\n')
    f.write("<thead>\n")
    f.write("<tr>\n")
    for col in json_file.columns:
        f.write(f"<th>{col}</th>\n")
    f.write("</tr>\n")
    f.write("</thead>\n")
    f.write("<tbody></tbody>\n")  # empty body for AJAX
    f.write("</table>\n")


def _get_table(files: list, file_table: str) -> str:
    return [f for f in files if f == file_table][0]
