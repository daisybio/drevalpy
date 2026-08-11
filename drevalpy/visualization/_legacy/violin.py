"""Plots a violin plot of the evaluation metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
from upath import UPath as Path

from ..requirements import PlotRequirement
from .vioheat import VioHeat

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult


class Violin(VioHeat):
    """Plots a violin plot of the evaluation metrics."""

    requirements: frozenset = frozenset({PlotRequirement.MULTIPLE_FOLDS})

    def __init__(
        self,
        result: ExperimentResult | None = None,
        *,
        df: pd.DataFrame | None = None,
        normalized_metrics: bool = False,
        whole_name: bool = False,
    ):
        """Initialize violin plot from evaluation results.

        :param result: Typed experiment result (preferred path).
        :param df: Legacy predictions DataFrame.
        :param normalized_metrics: Whether to show only normalized metric columns.
        :param whole_name: Whether to display full algorithm setting labels.
        """
        super().__init__(result=result, df=df, normalized_metrics=normalized_metrics, whole_name=whole_name)
        self.df["box"] = self.df["algorithm"] + "_" + self.df["rand_setting"] + "_" + self.df["test_mode"]
        self.df = self.df.dropna(axis=1, how="all")
        self.fig = go.Figure()
        self.occurring_metrics = [metric for metric in self.all_metrics if metric in self.df.columns]

    def draw_and_save(self, out_prefix: str | Path, out_suffix: str) -> None:
        """Draw violin plot and save as HTML.

        :param out_prefix: Output directory (for example ``results/my_run/violin_plots``).
        :param out_suffix: Filename suffix (for example ``algorithms_normalized``).
        """
        self._draw()
        path_out = Path(out_prefix) / f"violin_{out_suffix}.html"
        self.fig.write_html(path_out)

    def _draw(self) -> None:
        self._create_evaluation_violins()
        count_sum = (
            self.count_r2
            + self.count_pearson
            + self.count_spearman
            + self.count_kendall
            + self.count_mse
            + self.count_rmse
            + self.count_mae
        )
        buttons_update = list(
            [
                dict(
                    label="All Metrics",
                    method="update",
                    args=[
                        {"visible": [True] * count_sum},
                        {"title": "All Metrics"},
                    ],
                ),
                dict(
                    label="R^2",
                    method="update",
                    args=[
                        {"visible": [True] * self.count_r2 + [False] * (count_sum - self.count_r2)},
                        {"title": "R^2"},
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
                            + [False] * (count_sum - self.count_r2 - self.count_pearson - self.count_spearman)
                        },
                        {"title": "Spearman"},
                    ],
                ),
                dict(
                    label="Kendall",
                    method="update",
                    args=[
                        {
                            "visible": [False] * (self.count_r2 + self.count_pearson + self.count_spearman)
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
            ]
        )
        if not self.normalized_metrics:
            buttons_update += list(
                [
                    dict(
                        label="MSE",
                        method="update",
                        args=[
                            {
                                "visible": [False] * (count_sum - self.count_mse - self.count_rmse - self.count_mae)
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
                                "visible": [False] * (count_sum - self.count_rmse - self.count_mae)
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
                            {"visible": [False] * (count_sum - self.count_mae) + [True] * self.count_mae},
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

    def _create_evaluation_violins(self):
        print("Drawing Violin plots ...")
        self.count_r2 = 0
        self.count_pearson = 0
        self.count_spearman = 0
        self.count_kendall = 0
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
            elif "RMSE" in metric:
                self.count_rmse += 1 * len(self.df["box"].unique())
            elif "MSE" in metric:
                self.count_mse += 1 * len(self.df["box"].unique())
            elif "MAE" in metric:
                self.count_mae += 1 * len(self.df["box"].unique())
            self._add_violin(metric)

    def _add_violin(self, metric):
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
