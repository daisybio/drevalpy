import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Heatmap:
    def __init__(self, df: pd.DataFrame, normalized_metrics=False, whole_name=False):
        self.df = df.sort_index()
        # drop the columns that are not needed
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
        self.df = self.df[[col for col in self.df.columns if col in self.all_metrics]]
        self.normalized_metrics = normalized_metrics
        if self.normalized_metrics:
            self.df = self.df[[col for col in self.df.columns if "normalized" in col]]
            titles = [
                "Standard Errors over CV folds",
                "Mean R^2: normalized",
                "Mean Correlations: normalized",
            ]
            nr_subplots = 3
            self.plot_settings = ["standard_errors", "r2", "correlations"]
        else:
            self.df = self.df[
                [col for col in self.df.columns if "normalized" not in col]
            ]
            titles = [
                "Standard Errors over CV folds",
                "Mean R^2",
                "Mean Correlations",
                "Mean Errors",
            ]
            nr_subplots = 4
            self.plot_settings = ["standard_errors", "r2", "correlations", "errors"]
        self.whole_name = whole_name
        self.fig = make_subplots(rows=nr_subplots, cols=1, subplot_titles=tuple(titles))
        self.__draw_heatmap__()

    def __draw_heatmap__(self):
        print("Drawing heatmaps ...")
        for plot_setting in self.plot_settings:
            self.__draw_subplots__(plot_setting)
        self.fig.update_layout(
            height=1000, width=1100, title_text="Heatmap of the evaluation metrics"
        )
        self.fig.update_traces(showscale=False)

    def __draw_subplots__(self, plot_setting):
        idx_split = self.df.index.to_series().str.split("_")
        setting = idx_split.str[0:3].str.join("_")
        if plot_setting == "standard_errors":
            dt = self.df.groupby(setting).apply(
                lambda x: self.calc_summary_metric(x=x, std_error=True)
            )
            row_idx = 1
            colorscale = "Pinkyl"
        elif plot_setting == "r2":
            r2_columns = [col for col in self.df.columns if "R^2" in col]
            dt = self.df[r2_columns]
            dt = dt.groupby(setting).apply(
                lambda x: self.calc_summary_metric(x=x, std_error=False)
            )
            dt = dt.sort_values(by=r2_columns[0], ascending=True)
            row_idx = 2
            colorscale = "Blues"
        elif plot_setting == "correlations":
            corr_columns = [
                col
                for col in self.df.columns
                if "Pearson" in col
                or "Spearman" in col
                or "Kendall" in col
                or "Partial_Correlation" in col
            ]
            corr_columns.sort()
            dt = self.df[corr_columns]
            dt = dt.groupby(setting).apply(
                lambda x: self.calc_summary_metric(x=x, std_error=False)
            )
            dt = dt.sort_values(by=corr_columns[0], ascending=True)
            row_idx = 3
            colorscale = "Viridis"
        elif plot_setting == "errors":
            dt = self.df[["MSE", "RMSE", "MAE"]]
            dt = dt.groupby(setting).apply(
                lambda x: self.calc_summary_metric(x=x, std_error=False)
            )
            dt = dt.sort_values(by="MSE", ascending=False)
            row_idx = 4
            colorscale = "hot"
        else:
            raise ValueError("Unknown plot setting")
        if self.whole_name:
            labels = [i.replace("_", " ") for i in list(dt.index)]
        else:
            labels = [i.split("_")[0] for i in list(dt.index)]
        self.fig = self.fig.add_trace(
            go.Heatmap(
                z=dt.values,
                x=dt.columns,
                y=labels,
                colorscale=colorscale,
                texttemplate="%{z:.2f}",
            ),
            row=row_idx,
            col=1,
        )

    @staticmethod
    def calc_summary_metric(x, std_error=False):
        # make empty results series
        results = pd.Series(index=x.columns)
        # iterate over columns
        for col in x.columns:
            if np.count_nonzero(np.isnan(x[col])) == len(x[col]):
                results[col] = np.nan
            elif std_error:
                # calculate standard error
                results[col] = np.nanstd(x[col]) / np.sqrt(x.shape[0])
            else:
                # calculate mean
                results[col] = np.nanmean(x[col])
        return results
