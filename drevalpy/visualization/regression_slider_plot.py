"""Module for generating regression plots with a slider for Pearson correlation coefficient."""

from io import TextIOWrapper

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr

from ..pipeline_function import pipeline_function
from .outplot import OutPlot


class RegressionSliderPlot(OutPlot):
    """Generates regression plots with a slider for the Pearson correlation coefficient."""

    @pipeline_function
    def __init__(
        self,
        df: pd.DataFrame,
        lpo_lco_ldo: str,
        model: str,
        group_by: str = "drug",
        normalize=False,
    ):
        """
        Initialize the RegressionSliderPlot class.

        :param df: true vs. predicted values
        :param lpo_lco_ldo: setting, e.g., LPO
        :param model: model name
        :param group_by: either "drug" or "cell_line"
        :param normalize: whether to normalize the true and predicted values by the mean of the group
        """
        self.df = df[(df["LPO_LCO_LDO"] == lpo_lco_ldo) & (df["rand_setting"] == "predictions")]
        self.df = self.df[(self.df["algorithm"] == model)]
        self.group_by = group_by
        self.normalize = normalize
        self.fig = go.Figure()
        self.model = model

        if self.normalize:
            if self.group_by == "cell_line":
                self.df.loc[:, "y_true"] = self.df["y_true"] - self.df["mean_y_true_per_drug"]
                self.df.loc[:, "y_pred"] = self.df["y_pred"] - self.df["mean_y_true_per_drug"]
            else:
                self.df.loc[:, "y_true"] = self.df["y_true"] - self.df["mean_y_true_per_cell_line"]
                self.df.loc[:, "y_pred"] = self.df["y_pred"] - self.df["mean_y_true_per_cell_line"]

    @pipeline_function
    def draw_and_save(self, out_prefix: str, out_suffix: str) -> None:
        """
        Draw the regression plot and save it to a file.

        :param out_prefix: e.g., results/my_run/regression_plots/
        :param out_suffix: e.g., LPO_drug_SimpleNeuralNetwork
        """
        self._draw()
        self.fig.write_html(f"{out_prefix}regression_lines_{out_suffix}.html")

    def _draw(self):
        """Draw the regression plot."""
        print(f"Generating regression plots for {self.group_by}, normalize={self.normalize}...")
        self.df = self.df.groupby(self.group_by).filter(lambda x: len(x) > 1)
        pccs = self.df.groupby(self.group_by).apply(lambda x: pearsonr(x["y_true"], x["y_pred"])[0])
        pccs = pccs.reset_index()
        pccs.columns = [self.group_by, "pcc"]
        self.df = self.df.merge(pccs, on=self.group_by)
        self._render_plot()

    @staticmethod
    def write_to_html(lpo_lco_ldo: str, f: TextIOWrapper, *args, **kwargs) -> TextIOWrapper:
        """
        Write the plot to the final report file.

        :param lpo_lco_ldo: setting, e.g., LPO
        :param f: final report file
        :param args: additional arguments
        :param kwargs: additional keyword arguments, in this case all files
        :return: the final report file
        """
        files: list[str] = kwargs.get("files", [])
        f.write('<h2 id="regression_plots">Regression plots</h2>\n')
        f.write("<ul>\n")
        regr_files = [f for f in files if lpo_lco_ldo in f and f.startswith("regression_lines")]
        regr_files.sort()
        for regr_file in regr_files:
            f.write(f'<li><a href="regression_plots/{regr_file}" target="_blank">{regr_file}</a></li>\n')
        f.write("</ul>\n")
        return f

    def _render_plot(self):
        """Render the regression plot."""
        # sort df by group name
        df = self.df.sort_values(self.group_by)
        setting_title = self.model + " " + df["LPO_LCO_LDO"].unique()[0]
        if self.normalize:
            if self.group_by == "cell_line":
                setting_title += ", normalized by drug mean"
                hover_data = [
                    "pcc",
                    "cell_line",
                    "drug",
                    "mean_y_true_per_drug",
                    "algorithm",
                ]
            else:
                setting_title += ", normalized by cell line mean"
                hover_data = [
                    "pcc",
                    "cell_line",
                    "drug",
                    "mean_y_true_per_cell_line",
                    "algorithm",
                ]

        else:
            hover_data = ["pcc", "cell_line", "drug", "algorithm"]
        self.fig = px.scatter(
            df,
            x="y_true",
            y="y_pred",
            color=self.group_by,
            trendline="ols",
            hover_name=self.group_by,
            hover_data=hover_data,
            title=f"{setting_title}: Regression plot",
        )

        min_val = np.min([np.min(df["y_true"]), np.min(df["y_pred"])])
        max_val = np.max([np.max(df["y_true"]), np.max(df["y_pred"])])
        self.fig.update_xaxes(range=[min_val, max_val])
        self.fig.update_yaxes(range=[min_val, max_val])
        self._make_slider(setting_title)

    def _make_slider(self, setting_title: str) -> None:
        """
        Make a slider for the Pearson correlation coefficient.

        :param setting_title: title of the plot
        """
        n_ticks = 21
        steps = []
        # take the range from pcc (-1 - 1) and divide it into n_ticks-1 equal parts
        pcc_parts = np.linspace(-1, 1, n_ticks)
        for i in range(n_ticks):
            # from the fig data, get the hover data and check if it is greater than the pcc_parts[i]
            # only iterate over even numbers because there are scatter points and the ols line for each group
            pccs = [0 for _ in range(0, len(self.fig.data))]
            for j in range(0, len(self.fig.data)):
                if j % 2 == 0:
                    pccs[j] = self.fig.data[j].customdata[0, 0]
                else:
                    pccs[j] = self.fig.data[j - 1].customdata[0, 0]
            if i == n_ticks - 1:
                # last step
                visible_traces = pccs >= pcc_parts[i]
                title = (
                    f"{setting_title}: Slider for PCCs >= {str(round(pcc_parts[i], 1))} (step {str(i + 1)} "
                    f"of {str(n_ticks)})"
                )
            else:
                # get traces between pcc_parts[i] and pcc_parts[i+1]
                visible_traces_gt = pccs >= pcc_parts[i]
                visible_traces_lt = pccs < pcc_parts[i + 1]
                visible_traces = visible_traces_gt & visible_traces_lt
                title = (
                    f"{setting_title}: Slider for PCCs between {str(round(pcc_parts[i], 1))} "
                    f"and {str(round(pcc_parts[i + 1], 1))} (step {str(i + 1)} of {str(n_ticks)})"
                )
            step = dict(
                method="update",
                args=[{"visible": visible_traces}, {"title": title}],
                label=str(round(pcc_parts[i], 1)),
            )
            steps.append(step)

        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "Pearson correlation coefficient="},
                pad={"t": 50},
                steps=steps,
            )
        ]

        self.fig.update_layout(
            sliders=sliders,
            legend=dict(yanchor="top", y=1.0, xanchor="left", x=1.05),
        )
