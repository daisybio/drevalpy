import pandas as pd
import plotly.graph_objects as go


class Violin:
    def __init__(self, df: pd.DataFrame, normalized_metrics=False, whole_name=False):
        self.df = df.sort_index()
        self.df["box"] = (
            self.df["algorithm"]
            + "_"
            + self.df["rand_setting"]
            + "_"
            + self.df["LPO_LCO_LDO"]
        )
        # remove columns with only NaN values
        self.df = self.df.dropna(axis=1, how="all")
        self.normalized_metrics = normalized_metrics
        self.whole_name = whole_name
        self.fig = go.Figure()
        self.all_metrics = [
            "R^2",
            "R^2: drug normalized",
            "R^2: cell_line normalized",
            "Pearson",
            "Pearson: drug normalized",
            "Pearson: cell_line normalized",
            "Spearman",
            "Spearman: drug normalized",
            "Spearman: cell_line normalized",
            "Kendall",
            "Kendall: drug normalized",
            "Kendall: cell_line normalized",
            "Partial_Correlation",
            "Partial_Correlation: drug normalized",
            "Partial_Correlation: cell_line normalized",
            "MSE",
            "RMSE",
            "MAE",
        ]
        if self.normalized_metrics:
            self.all_metrics = [
                metric for metric in self.all_metrics if "normalized" in metric
            ]
        else:
            self.all_metrics = [
                metric for metric in self.all_metrics if "normalized" not in metric
            ]
        self.occurring_metrics = [
            metric for metric in self.all_metrics if metric in self.df.columns
        ]
        self.__draw_violin__()

    def __draw_violin__(self):
        self.__create_evaluation_violins__()
        count_sum = (
            self.count_r2
            + self.count_pearson
            + self.count_spearman
            + self.count_kendall
            + self.count_partial_correlation
            + self.count_mse
            + self.count_rmse
            + self.count_mae
        )
        buttons_update = list(
            [
                dict(
                    label="All Metrics",
                    method="update",
                    args=[{"visible": [True] * count_sum}, {"title": "All Metrics"}],
                ),
                dict(
                    label="R^2",
                    method="update",
                    args=[
                        {
                            "visible": [True] * self.count_r2
                            + [False] * (count_sum - self.count_r2)
                        },
                        {"title": "R^2"},
                    ],
                ),
                dict(
                    label="Correlations",
                    method="update",
                    args=[
                        {
                            "visible": [False] * self.count_r2
                            + [True]
                            * (
                                self.count_pearson
                                + self.count_spearman
                                + self.count_kendall
                                + self.count_partial_correlation
                            )
                            + [False]
                            * (self.count_mse + self.count_rmse + self.count_mae)
                        },
                        {"title": "All Correlations"},
                    ],
                ),
                dict(
                    label="Pearson",
                    method="update",
                    args=[
                        {
                            "visible": [False] * self.count_r2
                            + [True] * self.count_pearson
                            + [False] * (count_sum - self.count_r2 - self.count_pearson)
                        },
                        {"title": "Pearson"},
                    ],
                ),
                dict(
                    label="Spearman",
                    method="update",
                    args=[
                        {
                            "visible": [False] * (self.count_r2 + self.count_pearson)
                            + [True] * self.count_spearman
                            + [False]
                            * (
                                count_sum
                                - self.count_r2
                                - self.count_pearson
                                - self.count_spearman
                            )
                        },
                        {"title": "Spearman"},
                    ],
                ),
                dict(
                    label="Kendall",
                    method="update",
                    args=[
                        {
                            "visible": [False]
                            * (self.count_r2 + self.count_pearson + self.count_spearman)
                            + [True] * self.count_kendall
                            + [False]
                            * (
                                count_sum
                                - self.count_r2
                                - self.count_pearson
                                - self.count_spearman
                                - self.count_kendall
                            )
                        },
                        {"title": "Kendall"},
                    ],
                ),
                dict(
                    label="Partial Correlation",
                    method="update",
                    args=[
                        {
                            "visible": [False]
                            * (
                                count_sum
                                - self.count_partial_correlation
                                - self.count_mse
                                - self.count_rmse
                                - self.count_mae
                            )
                            + [True] * self.count_partial_correlation
                            + [False]
                            * (self.count_mse + self.count_rmse + self.count_mae)
                        },
                        {"title": "Partial Correlation"},
                    ],
                ),
            ]
        )
        if not self.normalized_metrics:
            buttons_update += list(
                [
                    dict(
                        label="Errors",
                        method="update",
                        args=[
                            {
                                "visible": [False]
                                * (
                                    count_sum
                                    - self.count_mse
                                    - self.count_rmse
                                    - self.count_mae
                                )
                                + [True]
                                * (self.count_mse + self.count_rmse + self.count_mae)
                            },
                            {"title": "All Errors"},
                        ],
                    ),
                    dict(
                        label="MSE",
                        method="update",
                        args=[
                            {
                                "visible": [False]
                                * (
                                    count_sum
                                    - self.count_mse
                                    - self.count_rmse
                                    - self.count_mae
                                )
                                + [True] * self.count_mse
                                + [False] * (self.count_rmse + self.count_mae)
                            },
                            {"title": "MSE"},
                        ],
                    ),
                    dict(
                        label="RMSE",
                        method="update",
                        args=[
                            {
                                "visible": [False]
                                * (count_sum - self.count_rmse - self.count_mae)
                                + [True] * self.count_rmse
                                + [False] * self.count_mae
                            },
                            {"title": "RMSE"},
                        ],
                    ),
                    dict(
                        label="MAE",
                        method="update",
                        args=[
                            {
                                "visible": [False] * (count_sum - self.count_mae)
                                + [True] * self.count_mae
                            },
                            {"title": "MAE"},
                        ],
                    ),
                ]
            )
        self.fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons_update,
                )
            ]
        )
        self.fig.update_layout(title_text="All Metrics", height=600, width=1100)

    def __create_evaluation_violins__(self):
        print("Drawing Violin plots ...")
        self.count_r2 = 0
        self.count_pearson = 0
        self.count_spearman = 0
        self.count_kendall = 0
        self.count_partial_correlation = 0
        self.count_mse = 0
        self.count_rmse = 0
        self.count_mae = 0
        for metric in self.occurring_metrics:
            if "R^2" in metric:
                self.count_r2 += 1 * len(self.df["box"].unique())
            elif "Pearson" in metric:
                self.count_pearson += 1 * len(self.df["box"].unique())
            elif "Spearman" in metric:
                self.count_spearman += 1 * len(self.df["box"].unique())
            elif "Kendall" in metric:
                self.count_kendall += 1 * len(self.df["box"].unique())
            elif "Partial_Correlation" in metric:
                self.count_partial_correlation += 1 * len(self.df["box"].unique())
            elif "RMSE" in metric:
                self.count_rmse += 1 * len(self.df["box"].unique())
            elif "MSE" in metric:
                self.count_mse += 1 * len(self.df["box"].unique())
            elif "MAE" in metric:
                self.count_mae += 1 * len(self.df["box"].unique())
            self.__add_violin__(metric)

    def __add_violin__(self, metric):
        for box in self.df["box"].unique():
            tmp_df = self.df[self.df["box"] == box]
            if self.whole_name:
                label = box + ": " + metric
            else:
                label = box.split("_")[0] + ": " + metric
            self.fig.add_trace(
                go.Violin(
                    y=tmp_df[metric],
                    x=[label] * len(tmp_df[metric]),
                    name=label,
                    box_visible=True,
                    meanline_visible=True,
                )
            )
