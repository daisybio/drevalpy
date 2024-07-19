from typing import TextIO, List


class HTMLTable:
    def __init__(self, df, export_path, grouping):
        self.df = df
        self.export_path = export_path
        self.grouping = grouping
        self.__create_html_table__()

    @staticmethod
    def write_to_html(lpo_lco_ldo: str, f: TextIO, files: List) -> TextIO:
        f.write('<h2 id="tables"> Evaluation Results Table</h2>\n')
        whole_table = HTMLTable.__get_table__(
            files=files, file_table=f"table_{lpo_lco_ldo}.html"
        )
        HTMLTable.__write_table__(f=f, table=whole_table)

        if lpo_lco_ldo != "LCO":
            f.write("<h2> Evaluation Results per Cell Line Table</h2>\n")
            cell_line_table = HTMLTable.__get_table__(
                files=files, file_table=f"table_{lpo_lco_ldo}_per_cl.html"
            )
            HTMLTable.__write_table__(f=f, table=cell_line_table)
        if lpo_lco_ldo != "LDO":
            f.write("<h2> Evaluation Results per Drug Table</h2>\n")
            drug_table = HTMLTable.__get_table__(
                files=files, file_table=f"table_{lpo_lco_ldo}_per_drug.html"
            )
            HTMLTable.__write_table__(f=f, table=drug_table)
        return f

    @staticmethod
    def __write_table__(f: TextIO, table: str):
        with open(table, "r") as eval_f:
            eval_results = eval_f.readlines()
            eval_results[0] = eval_results[0].replace(
                '<table border="1" class="dataframe">',
                '<table class="display customDataTable" style="width:100%">',
            )
            for line in eval_results:
                f.write(line)

    @staticmethod
    def __get_table__(files: List, file_table: str) -> str:
        return [f for f in files if f == file_table][0]

    def __create_html_table__(self):
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
        if self.grouping == "drug":
            selected_columns = ["drug"] + selected_columns
        elif self.grouping == "cell_line":
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

    def export_html_table(self):
        self.df.to_html(self.export_path, index=False)
