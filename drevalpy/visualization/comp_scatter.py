"""Contains the code needed to draw the correlation comparison scatter plot."""

from io import TextIOWrapper

import pandas as pd
import plotly.graph_objects as go

from ..models import SINGLE_DRUG_MODEL_FACTORY
from .outplot import OutPlot


class ComparisonScatter(OutPlot):
    """
    Class to draw scatter plots for comparison of correlation metrics between models.

    Produces two types of plots: an overall comparison plot and a dropdown plot for comparison between all models.
    If one model is consistently better than the other, the points deviate from the identity line (higher if the
    model is on the y-axis, lower if it is on the x-axis.
    The dropdown plot allows to select two models for comparison of their per-drug/per-cell-line pearson correlation.
    The overall plot facets all models and visualizes the density of the points.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        color_by: str,
        test_mode: str,
        metric: str = "R^2",
        algorithm: str = "all",
    ):
        """
        Initialize the ComparisonScatter object.

        :param df: evaluation results per group, either drug or cell line
        :param color_by: group variable, i.e., drug or cell line
        :param test_mode: evaluation test_mode, e.g., LCO (leave-cell-line-out)
        :param metric: correlation metric to be compared. Default is Pearson.
        :param algorithm: used to distinguish between per-algorithm plots and per-test_mode plots (all models then).
        """
        exclude_models = (
            {"NaiveDrugMeanPredictor"}.union({model for model in SINGLE_DRUG_MODEL_FACTORY.keys()})
            if color_by == "drug"
            else {"NaiveCellLineMeanPredictor"}
        )
        exclude_models.add("NaivePredictor")
        exclude_models.add("NaiveMeanEffectsPredictor")

        self.df = df.sort_values("model")
        self.name: str | None = None
        if algorithm == "all":
            # draw plots for comparison between all models
            self.df = self.df[
                (self.df["test_mode"] == test_mode)
                & (self.df["rand_setting"] == "predictions")
                & (~self.df["algorithm"].isin(exclude_models))
                # and exclude all lines for which algorithm starts with any element from
                # exclude_models
                & (~self.df["algorithm"].str.startswith(tuple(exclude_models)))
            ]
            self.name = f"{color_by}_{test_mode}"
        elif algorithm not in exclude_models:
            # draw plots for comparison between all test settings of one model
            self.df = self.df[(self.df["test_mode"] == test_mode) & (self.df["algorithm"] == algorithm)]
            self.name = f"{color_by} {algorithm} {test_mode}"
        if self.df.empty:
            print(f"No data found for {self.name}. Skipping ...")
            return
        self.color_by = color_by
        self.metric = metric

        self.df["test_mode"] = self.df["model"].str.split("_").str[0:3].str.join("_")
        self.models = self.df["test_mode"].unique()

        self.dropdown_fig = go.Figure()
        self.dropdown_buttons_x: list[dict] = list()
        self.dropdown_buttons_y: list[dict] = list()

    def draw_and_save(self, out_prefix: str, out_suffix: str) -> None:
        """
        Draws and saves the scatter plots.

        :param out_prefix: e.g., results/my_run/comp_scatter/
        :param out_suffix: should be self.name
        :raises AssertionError: if out_suffix does not match self.name
        """
        if self.df.empty:
            return
        self._draw()
        if self.name != out_suffix:
            raise AssertionError(f"Name mismatch: {self.name} != {out_suffix}")
        path_out = f"{out_prefix}comp_scatter_{out_suffix}.html"
        self.dropdown_fig.write_html(path_out)

    def _draw(self) -> None:
        """Draws the scatter plots."""
        print("Drawing scatterplots ...")
        self._generate_comp_scatterplots()

        self.dropdown_fig.update_layout(
            title=f'{str(self.color_by).replace("_", " ").capitalize()}-wise scatter plot of {self.metric} '
            f"for each model",
            showlegend=False,
        )
        # Set dropdown menu
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
    def write_to_html(test_mode: str, f: TextIOWrapper, *args, **kwargs) -> TextIOWrapper:
        """
        Inserts the generated files into the result HTML file.

        :param test_mode: test_mode, e.g., LCO
        :param f: file to write to
        :param args: unused
        :param kwargs: used to get all files generated by create_report / the pipeline
        :returns: the file f
        """
        files: list[str] = kwargs.get("files", [])
        f.write('<h2 id="corr_comp">Comparison of normalized R^2 values</h2>\n')
        f.write(
            "R^2 values can be compared here between models, either per cell line or per drug. "
            "This can either show if a model has consistently higher or lower R^2 values than another model or "
            "identify cell lines/drugs for which models agree or disagree.\n"
            "The x-axis is the first dropdown menu, the y-axis is the second dropdown menu.\n"
        )
        for group_by in ["drug_name", "cell_line_name"]:
            plot_list = [f for f in files if f.startswith("comp_scatter") and f.endswith(f"{test_mode}.html")]
            if f"comp_scatter_{group_by}_{test_mode}.html" in plot_list:
                f.write(f'<h3 id="corr_comp_drug">{group_by.capitalize()}-wise comparison</h3>\n')
                f.write(
                    f'<iframe src="comp_scatter/comp_scatter_{group_by}_{test_mode}.html" '
                    f'width="100%" height="100%" frameBorder="0"></iframe>\n'
                )
                f.write("<h4>Comparisons per model</h4>\n")
                f.write("<ul>\n")
                listed_files = [
                    elem
                    for elem in plot_list
                    if (
                        elem != f"comp_scatter_{group_by}_{test_mode}.html"
                        and elem != f"comp_scatter_overall_{group_by}_{test_mode}.html"
                    )
                ]
                listed_files.sort()
                for group_comparison in listed_files:
                    f.write(
                        f'<li><a href="comp_scatter/{group_comparison}" target="_blank">'
                        f"{group_comparison}</a></li>\n"
                    )
                f.write("</ul>\n")
        return f

    def _generate_comp_scatterplots(self) -> None:
        """Generates the scatter plots."""
        # render first scatterplot that is shown in the dropdown plot
        first_df = self._subset_df(run_id=self.models[0])
        if self.color_by == "drug_name":
            hover_variables = ["drug_name", "pubchem_id"]
        else:
            hover_variables = ["cell_line_name", "cellosaurus_id"]
        scatterplot = go.Scatter(
            x=first_df[self.metric],
            y=first_df[self.metric],
            mode="markers",
            marker=dict(size=6, showscale=False),
            customdata=first_df[hover_variables],
            hovertemplate="<br>".join(
                [
                    f"{self.color_by.capitalize()}: %{{customdata[0]}}",
                    f"{hover_variables[1]}: %{{customdata[1]}}",
                    "x: %{{x:.2f}}",
                    "y: %{{y:.2f}}",
                ]
            ),
            showlegend=True,
            visible=True,
        )
        self.dropdown_fig.add_trace(scatterplot)

        for run_idx in range(len(self.models)):
            run = self.models[run_idx]
            x_df = self._subset_df(run_id=run)
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
                y_df = self._subset_df(run_id=run2)

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

    def _subset_df(self, run_id: str) -> pd.DataFrame:
        """
        Subsets the dataframe for a given run_id to the relevant columns and sets the index to the color_by variable.

        :param run_id: user-defined ID of the whole run
        :returns: subsetted dataframe
        """
        subset_cols = [self.metric, self.color_by, "model"]
        if self.color_by == "drug_name":
            subset_cols.append("pubchem_id")
        else:
            subset_cols.append("cellosaurus_id")
        s_df = self.df[self.df["test_mode"] == run_id][subset_cols]
        # sort by color_by variable
        s_df = s_df.sort_values(self.color_by)
        s_df[self.metric] = s_df[self.metric].fillna(0)
        return s_df
