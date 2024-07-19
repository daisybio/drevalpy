from typing import TextIO, List

import pandas as pd

from drevalpy.visualization.outplot import OutPlot


class HTMLTable(OutPlot):
    def __init__(self, df: pd.DataFrame, group_by: str):
        self.df = df
        self.group_by = group_by

    def draw_and_save(self, out_prefix: str, out_suffix: str) -> None:
        self.__draw__()
        path_out = f"{out_prefix}table_{out_suffix}.html"
        self.df.to_html(path_out, index=False)

    def __draw__(self) -> None:
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
            "Partial_Correlation",
            "LPO_LCO_LDO",
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
                "R^2: drug normalized",
                "Pearson: drug normalized",
                "R^2: cell_line normalized",
                "Pearson: cell_line normalized",
                "RMSE",
                "MAE",
                "Spearman",
                "Kendall",
                "Partial_Correlation",
                "Spearman: drug normalized",
                "Kendall: drug normalized",
                "Partial_Correlation: drug normalized",
                "Spearman: cell_line normalized",
                "Kendall: cell_line normalized",
                "Partial_Correlation: cell_line normalized",
                "LPO_LCO_LDO",
            ]
        # reorder columns
        self.df = self.df[selected_columns]

    @staticmethod
    def write_to_html(lpo_lco_ldo: str, f: TextIO, *args, **kwargs) -> TextIO:
        files = kwargs.get("files")
        f.write('<h2 id="tables"> Evaluation Results Table</h2>\n')
        whole_table = __get_table__(files=files, file_table=f"table_{lpo_lco_ldo}.html")
        __write_table__(f=f, table=whole_table)

        if lpo_lco_ldo != "LCO":
            f.write("<h2> Evaluation Results per Cell Line Table</h2>\n")
            cell_line_table = __get_table__(
                files=files, file_table=f"table_{lpo_lco_ldo}_per_cl.html"
            )
            __write_table__(f=f, table=cell_line_table)
        if lpo_lco_ldo != "LDO":
            f.write("<h2> Evaluation Results per Drug Table</h2>\n")
            drug_table = __get_table__(
                files=files, file_table=f"table_{lpo_lco_ldo}_per_drug.html"
            )
            __write_table__(f=f, table=drug_table)
        return f


def __write_table__(f: TextIO, table: str):
    with open(table, "r") as eval_f:
        eval_results = eval_f.readlines()
        eval_results[0] = eval_results[0].replace(
            '<table border="1" class="dataframe">',
            '<table class="display customDataTable" style="width:100%">',
        )
        for line in eval_results:
            f.write(line)


def __get_table__(files: List, file_table: str) -> str:
    return [f for f in files if f == file_table][0]
