from typing import TextIO
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from drevalpy.visualization.outplot import OutPlot
from drevalpy.models import SINGLE_DRUG_MODEL_FACTORY


class CorrelationComparisonScatter(OutPlot):
    def __init__(
        self,
        df: pd.DataFrame,
        color_by: str,
        lpo_lco_ldo: str,
        metric="Pearson",
        algorithm="all",
    ):
        exclude_models = (
            {"NaiveDrugMeanPredictor"}.union(
                {model for model in SINGLE_DRUG_MODEL_FACTORY.keys()}
            )
            if color_by == "drug"
            else {"NaiveCellLineMeanPredictor"}
        )
        exclude_models.add("NaivePredictor")

        self.df = df.sort_values("model")
        if algorithm == "all":
            # draw plots for comparison between all models
            self.df = self.df[
                (self.df["LPO_LCO_LDO"] == lpo_lco_ldo)
                & (self.df["rand_setting"] == "predictions")
                & (~self.df["algorithm"].isin(exclude_models))
                # and exclude all lines for which algorithm starts with any element from
                # exclude_models
                & (~self.df["algorithm"].str.startswith(tuple(exclude_models)))
            ]
            self.name = f"{color_by}_{lpo_lco_ldo}"
        elif algorithm not in exclude_models:
            # draw plots for comparison between all test settings of one model
            self.df = self.df[
                (self.df["LPO_LCO_LDO"] == lpo_lco_ldo)
                & (self.df["algorithm"] == algorithm)
            ]
            self.name = f"{color_by} {algorithm} {lpo_lco_ldo}"
        else:
            self.name = None
        if self.df.empty:
            print(f"No data found for {self.name}. Skipping ...")
            return
        self.color_by = color_by
        self.metric = metric

        self.df["setting"] = self.df["model"].str.split("_").str[0:3].str.join("_")
        self.models = self.df["setting"].unique()

        self.fig_overall = make_subplots(
            rows=len(self.models),
            cols=len(self.models),
            subplot_titles=[
                str(model).replace("_", "<br>", 2) for model in self.models
            ],
        )

        # Update axis labels
        for i in range(len(self.models)):
            for j in range(len(self.models)):
                self.fig_overall.update_xaxes(
                    title_text=f"{self.models[j].split('_')[0]} {metric} Score",
                    row=i + 1,
                    col=j + 1,
                )
                self.fig_overall.update_yaxes(
                    title_text=f"{self.models[i].split('_')[0]} {metric} Score",
                    row=i + 1,
                    col=j + 1,
                )

        for i in range(len(self.models)):
            self.fig_overall["layout"]["annotations"][i]["font"]["size"] = 12
        self.dropdown_fig = go.Figure()
        self.dropdown_buttons_x = list()
        self.dropdown_buttons_y = list()

    def draw_and_save(self, out_prefix: str, out_suffix: str) -> None:
        if self.df.empty:
            return
        self.__draw__()
        assert self.name == out_suffix
        path_out = f"{out_prefix}corr_comp_scatter_{out_suffix}.html"
        self.dropdown_fig.write_html(path_out)
        path_out = f"{out_prefix}corr_comp_scatter_overall_{out_suffix}.html"
        self.fig_overall.write_html(path_out)

    def __draw__(self) -> None:
        print("Drawing scatterplots ...")
        self.__generate_corr_comp_scatterplots__()
        self.fig_overall.update_layout(
            title=f'{str(self.color_by).replace("_", " ").capitalize()}-wise scatter plot of {self.metric} for each model',
            showlegend=False,
        )
        self.dropdown_fig.update_layout(
            title=f'{str(self.color_by).replace("_", " ").capitalize()}-wise scatter plot of {self.metric} for each model',
            showlegend=False,
        )
        self.dropdown_fig.update_layout(
            updatemenus=[
                {
                    "buttons": self.dropdown_buttons_x,
                    "direction": "down",
                    "showactive": True,
                    "x": 0.0,
                    "xanchor": "left",
                    "y": 1.5,
                    "yanchor": "top",
                },
                {
                    "buttons": self.dropdown_buttons_y,
                    "direction": "down",
                    "showactive": True,
                    "x": 0.5,
                    "xanchor": "left",
                    "y": 1.5,
                    "yanchor": "top",
                },
            ]
        )
        self.dropdown_fig.update_xaxes(range=[-1, 1])
        self.dropdown_fig.update_yaxes(range=[-1, 1])

    @staticmethod
    def write_to_html(lpo_lco_ldo: str, f: TextIO, *args, **kwargs) -> TextIO:
        files = kwargs.get("files")
        f.write('<h2 id="corr_comp">Comparison of correlation metrics</h2>\n')
        for group_by in ["drug", "cell_line"]:
            plot_list = [
                f
                for f in files
                if f.startswith("corr_comp_scatter")
                and f.endswith(f"{lpo_lco_ldo}.html")
            ]
            if f"corr_comp_scatter_{group_by}_{lpo_lco_ldo}.html" in plot_list:
                f.write(
                    f'<h3 id="corr_comp_drug">{group_by.capitalize()}-wise comparison</h3>\n'
                )
                f.write("<h4>Overall comparison between models</h4>\n")
                f.write(
                    f'<iframe src="corr_comp_scatter/corr_comp_scatter_overall_{group_by}_{lpo_lco_ldo}.html" '
                    f'width="100%" height="100%" frameBorder="0"></iframe>\n'
                )
                f.write("<h4>Comparison between all models, dropdown menu</h4>\n")
                f.write(
                    f'<iframe src="corr_comp_scatter/corr_comp_scatter_{group_by}_{lpo_lco_ldo}.html" '
                    f'width="100%" height="100%" frameBorder="0"></iframe>\n'
                )
                f.write("<h4>Comparisons per model</h4>\n")
                f.write("<ul>\n")
                listed_files = [
                    elem
                    for elem in plot_list
                    if elem != f"corr_comp_scatter_{lpo_lco_ldo}_{group_by}.html"
                    and elem
                    != f"corr_comp_scatter_overall_{lpo_lco_ldo}_{group_by}.html"
                ]
                listed_files.sort()
                for group_comparison in listed_files:
                    f.write(
                        f'<li><a href="corr_comp_scatter/{group_comparison}" target="_blank">'
                        f"{group_comparison}</a></li>\n"
                    )
                f.write("</ul>\n")
        return f

    def __generate_corr_comp_scatterplots__(self):
        # render first scatterplot that is shown in the dropdown plot
        first_df = self.__subset_df__(run_id=self.models[0])
        scatterplot = go.Scatter(
            x=first_df[self.metric],
            y=first_df[self.metric],
            mode="markers",
            marker=dict(size=6, showscale=False),
            text=first_df.index,
            showlegend=True,
            visible=True,
        )
        self.dropdown_fig.add_trace(scatterplot)

        # identity line shown in every subplot of the static scatterplot
        line_corr = go.Scatter(
            x=[-1, 1],
            y=[-1, 1],
            mode="lines",
            line=dict(color="gold", width=2, dash="dash"),
            showlegend=False,
            visible=True,
        )

        for run_idx in range(len(self.models)):
            run = self.models[run_idx]
            x_df = self.__subset_df__(run_id=run)
            self.dropdown_buttons_x.append(
                dict(
                    label=run,
                    method="update",
                    args=[
                        {"x": [x_df[self.metric]]},
                        {"xaxis": {"title": run, "range": [-1, 1]}},
                    ],
                )
            )
            for run2_idx in range(len(self.models)):
                run2 = self.models[run2_idx]
                y_df = self.__subset_df__(run_id=run2)

                scatterplot = self.__draw_subplot__(x_df, y_df, run, run2)
                self.fig_overall.add_trace(
                    scatterplot, col=run_idx + 1, row=run2_idx + 1
                )
                self.fig_overall.add_trace(line_corr, col=run_idx + 1, row=run2_idx + 1)

                # create dropdown buttons for y axis only in the first iteration
                if run_idx == 0:
                    self.dropdown_buttons_y.append(
                        dict(
                            label=run2,
                            method="update",
                            args=[
                                {"y": [y_df[self.metric]]},
                                {"yaxis": {"title": run2, "range": [-1, 1]}},
                            ],
                        )
                    )
                    # set y axis title
                    if run2_idx == 0:
                        self.fig_overall["layout"]["yaxis"]["title"] = str(
                            run2
                        ).replace("_", "<br>", 2)
                        self.fig_overall["layout"]["yaxis"]["title"]["font"]["size"] = 6
                    else:
                        y_axis_idx = (run2_idx) * len(self.models) + 1
                        self.fig_overall["layout"][f"yaxis{y_axis_idx}"]["title"] = str(
                            run2
                        ).replace("_", "<br>", 2)
                        self.fig_overall["layout"][f"yaxis{y_axis_idx}"]["title"][
                            "font"
                        ]["size"] = 6

    def __subset_df__(self, run_id: str):
        s_df = self.df[self.df["setting"] == run_id][
            [self.metric, self.color_by, "model"]
        ]
        s_df.set_index(self.color_by, inplace=True)
        s_df.sort_index(inplace=True)
        s_df[self.metric] = s_df[self.metric].fillna(0)
        return s_df

    def __draw_subplot__(self, x_df, y_df, run, run2):
        # only retain the common indices
        common_indices = x_df.index.intersection(y_df.index)
        x_df_inter = x_df.loc[common_indices]
        y_df = y_df.loc[common_indices]
        x_df_inter["setting"] = x_df_inter["model"].str.split("_").str[4:].str.join("")
        y_df["setting"] = y_df["model"].str.split("_").str[4:].str.join("")

        joint_df = pd.concat([x_df_inter, y_df], axis=1)
        joint_df.columns = [
            f"{self.metric}_x",
            "model_x",
            "setting_x",
            f"{self.metric}_y",
            "model_y",
            "setting_y",
        ]

        density = self.__get_density__(
            joint_df[f"{self.metric}_x"], joint_df[f"{self.metric}_y"]
        )
        joint_df["color"] = density

        custom_text = joint_df.apply(
            lambda row: f"<i>{self.color_by.capitalize()}:</i>: {row.name}<br>"
            + f"<i>Split x:</i>: {row.setting_x}<br>"
            + f"<i>Split y:</i>: {row.setting_y}<br>",
            axis=1,
        )

        scatterplot = go.Scatter(
            x=x_df_inter[self.metric],
            y=y_df[self.metric],
            mode="markers",
            marker=dict(size=4, color=density, colorscale="Viridis", showscale=False),
            showlegend=False,
            visible=True,
            meta=[run, run2],
            text=custom_text,
        )
        return scatterplot

    @staticmethod
    def __get_density__(x: pd.Series, y: pd.Series):
        """Get kernal density estimate for each (x, y) point."""
        try:
            values = np.vstack([x, y])
            kernel = stats.gaussian_kde(values)
            density = kernel(values)
        except scipy.linalg.LinAlgError:
            density = np.zeros(len(x))
        return density
