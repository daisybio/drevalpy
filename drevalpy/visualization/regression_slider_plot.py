import plotly.express as px
from scipy.stats import pearsonr
import numpy as np
import pandas as pd


class RegressionSliderPlot:
    def __init__(self, df: pd.DataFrame, group_by: str = "drug", normalize=False):
        self.df = df[df["rand_setting"] == "predictions"]
        self.group_by = group_by
        self.normalize = normalize
        self.fig = None

        if self.normalize:
            if self.group_by == "cell_line":
                self.df["y_true"] = self.df["y_true"] - self.df["mean_y_true_per_drug"]
                self.df["y_pred"] = self.df["y_pred"] - self.df["mean_y_true_per_drug"]
            else:
                self.df["y_true"] = (
                    self.df["y_true"] - self.df["mean_y_true_per_cell_line"]
                )
                self.df["y_pred"] = (
                    self.df["y_pred"] - self.df["mean_y_true_per_cell_line"]
                )

        self.__draw_regression_plot__()

    def __draw_regression_plot__(self):
        print(
            f"Generating regression plots for {self.group_by}, normalize={self.normalize}..."
        )
        self.df = self.df.groupby(self.group_by).filter(lambda x: len(x) > 1)
        pccs = self.df.groupby(self.group_by).apply(
            lambda x: pearsonr(x["y_true"], x["y_pred"])[0]
        )
        pccs = pccs.reset_index()
        pccs.columns = [self.group_by, "pcc"]
        self.df = self.df.merge(pccs, on=self.group_by)
        self.__render_plot__()

    def __render_plot__(self):
        # sort df by group name
        df = self.df.sort_values(self.group_by)
        setting_title = (
            df["algorithm"].unique()[0] + " " + df["LPO_LCO_LDO"].unique()[0]
        )
        if self.normalize:
            if self.group_by == "cell_line":
                setting_title += f", normalized by drug mean"
                hover_data = ["pcc", "cell_line", "drug", "mean_y_true_per_drug"]
            else:
                setting_title += f", normalized by cell line mean"
                hover_data = ["pcc", "cell_line", "drug", "mean_y_true_per_cell_line"]

        else:
            hover_data = ["pcc", "cell_line", "drug"]
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
        self.__make_slider__(setting_title)

    def __make_slider__(self, setting_title):
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
                title = f"{setting_title}: Slider for PCCs >= {str(round(pcc_parts[i], 1))} (step {str(i + 1)} of {str(n_ticks)})"
            else:
                # get traces between pcc_parts[i] and pcc_parts[i+1]
                visible_traces_gt = pccs >= pcc_parts[i]
                visible_traces_lt = pccs < pcc_parts[i + 1]
                visible_traces = visible_traces_gt & visible_traces_lt
                title = f"{setting_title}: Slider for PCCs between {str(round(pcc_parts[i], 1))} and {str(round(pcc_parts[i + 1], 1))} (step {str(i + 1)} of {str(n_ticks)})"
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
            sliders=sliders, legend=dict(yanchor="top", y=1.0, xanchor="left", x=1.05)
        )
