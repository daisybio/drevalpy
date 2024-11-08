"""Plots a heatmap of the evaluation metrics."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..pipeline_function import pipeline_function
from .vioheat import VioHeat


class Heatmap(VioHeat):
    """Plots a heatmap of the evaluation metrics."""

    @pipeline_function
    def __init__(self, df: pd.DataFrame, normalized_metrics=False, whole_name=False):
        """
        Initialize the Heatmap class.

        :param df: either containing all predictions for all algorithms or all tests for one algorithm (including
            robustness, randomization, … tests then)
        :param normalized_metrics: whether the metrics are normalized
        :param whole_name: whether the whole name should be displayed
        """
        super().__init__(df, normalized_metrics, whole_name)
        self.df = self.df[[col for col in self.df.columns if col in self.all_metrics]]
        if self.normalized_metrics:
            titles = [
                "Standard Errors over CV folds",
                "Mean R^2: normalized",
                "Mean Correlations: normalized",
            ]
            nr_subplots = 3
            self.plot_settings = ["standard_errors", "r2", "correlations"]
            self.fig = make_subplots(
                rows=nr_subplots,
                cols=1,
                subplot_titles=tuple(titles),
                vertical_spacing=0.25,
            )
        else:
            titles = [
                "Standard Errors over CV folds",
                "Mean R^2",
                "Mean Correlations",
                "Mean Errors",
            ]
            nr_subplots = 4
            self.plot_settings = [
                "standard_errors",
                "r2",
                "correlations",
                "errors",
            ]
            self.fig = make_subplots(
                rows=nr_subplots,
                cols=1,
                subplot_titles=tuple(titles),
                vertical_spacing=0.1,
            )

    @pipeline_function
    def draw_and_save(self, out_prefix: str, out_suffix: str) -> None:
        """
        Draw the heatmap and save it to a file.

        :param out_prefix: e.g., results/my_run/heatmaps/
        :param out_suffix: e.g., algorithms_normalized
        """
        self._draw()
        path_out = f"{out_prefix}heatmap_{out_suffix}.html"
        self.fig.write_html(path_out)

    def _draw(self) -> None:
        """Draw the heatmap."""
        print("Drawing heatmaps ...")
        for plot_setting in self.plot_settings:
            self._draw_subplots(plot_setting)
        self.fig.update_layout(
            height=1000,
            width=1100,
            title_text="Heatmap of the evaluation metrics",
        )
        self.fig.update_traces(showscale=False)

    def _draw_subplots(self, plot_setting: str) -> None:
        """
        Draw the subplots of the heatmap.

        :param plot_setting: Either "standard_errors", "r2", "correlations", or "errors"
        :raises ValueError: If an unknown plot setting is given
        """
        idx_split = self.df.index.to_series().str.split("_")
        setting = idx_split.str[0:3].str.join("_")
        if plot_setting == "standard_errors":
            dt = self.df.groupby(setting).apply(lambda x: self._calc_summary_metric(x=x, std_error=True))
            row_idx = 1
            colorscale = "Pinkyl"
        elif plot_setting == "r2":
            r2_columns = [col for col in self.df.columns if "R^2" in col]
            dt = self.df[r2_columns]
            dt = dt.groupby(setting).apply(lambda x: self._calc_summary_metric(x=x, std_error=False))
            dt = dt.sort_values(by=r2_columns[0], ascending=True)
            row_idx = 2
            colorscale = "Blues"
        elif plot_setting == "correlations":
            corr_columns = [
                col
                for col in self.df.columns
                if "Pearson" in col or "Spearman" in col or "Kendall" in col or "Partial_Correlation" in col
            ]
            corr_columns.sort()
            dt = self.df[corr_columns]
            dt = dt.groupby(setting).apply(lambda x: self._calc_summary_metric(x=x, std_error=False))
            dt = dt.sort_values(by=corr_columns[0], ascending=True)
            row_idx = 3
            colorscale = "Viridis"
        elif plot_setting == "errors":
            dt = self.df[["MSE", "RMSE", "MAE"]]
            dt = dt.groupby(setting).apply(lambda x: self._calc_summary_metric(x=x, std_error=False))
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
    def _calc_summary_metric(x: pd.DataFrame, std_error: bool = False):
        """
        Calculate the mean or standard error of the metrics.

        :param x: DataFrame containing the metrics
        :param std_error: whether to calculate the standard error or the mean
        :returns: Series containing the mean or standard error of the metrics
        """
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
