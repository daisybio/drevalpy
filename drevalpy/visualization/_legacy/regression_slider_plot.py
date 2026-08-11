"""Module for generating regression plots with a slider for Pearson correlation coefficient."""

from __future__ import annotations

from io import TextIOWrapper
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr
from upath import UPath as Path

from .outplot import OutPlot

if TYPE_CHECKING:
    from drevalpy.types.results import ModelResult


def _build_regression_df_from_model(result: ModelResult) -> pd.DataFrame:
    """Build a flat DataFrame from a ModelResult for scatter regression plots.

    Columns: y_true, y_pred, algorithm, test_mode, rand_setting, CV_split, drug_ids, cell_line_ids.
    """
    rows: list[dict] = []
    for run in result.runs:
        if run.randomization is not None:
            continue
        for i in range(len(run.predictions)):
            rows.append(
                {
                    "y_true": float(run.ground_truth[i]),
                    "y_pred": float(run.predictions[i]),
                    "algorithm": run.model_name,
                    "test_mode": run.split_mode,
                    "rand_setting": "predictions",
                    "CV_split": run.fold_index,
                    "drug_name": str(run.drug_ids[i]),
                    "cell_line_name": str(run.cell_line_ids[i]),
                }
            )
    return pd.DataFrame(rows)


class RegressionSliderPlot(OutPlot):
    """Generates regression plots with a slider for the Pearson correlation coefficient."""

    result_type: str = "ModelResult"
    requirements: frozenset = frozenset()

    def __init__(
        self,
        result: ModelResult | None = None,
        *,
        df: pd.DataFrame | None = None,
        test_mode: str = "",
        model: str = "",
        group_by: str = "drug_name",
        normalize: bool = False,
    ):
        """Initialize regression slider plot.

        :param result: Typed model result (preferred path).
        :param df: Legacy true-vs-predicted values table.
        :param test_mode: Evaluation test mode (for example ``"LPO"``).
        :param model: Model name to plot.
        :param group_by: Grouping column (``"drug_name"`` or ``"cell_line_name"``).
        :param normalize: Subtract NaiveMeanEffectsPredictor predictions per pair.
        """
        if result is not None:
            self.df = _build_regression_df_from_model(result)
            self.model = result.model_name
            self.group_by = group_by
            self.normalize = normalize
            self.fig = go.Figure()
            return

        if df is None:
            raise ValueError("Either 'result' or 'df' must be provided")

        self.df = df[(df["test_mode"] == test_mode) & (df["rand_setting"] == "predictions")]
        model_df = self.df[(self.df["algorithm"] == model)]
        self.df = model_df
        self.group_by = group_by
        self.normalize = normalize
        self.fig = go.Figure()
        self.model = model

        if self.normalize:
            mean_effects_df = df[
                (df["algorithm"] == "NaiveMeanEffectsPredictor")
                & (df["test_mode"] == test_mode)
                & (df["rand_setting"] == "predictions")
            ]
            merged_df = model_df.merge(
                mean_effects_df,
                on=["pubchem_id", "drug_name", "cellosaurus_id", "cell_line_name", "rand_setting", "test_mode"],
                how="left",
            )
            merged_df.loc[:, "y_true"] = merged_df["y_true_x"] - merged_df["y_pred_y"]
            merged_df.loc[:, "y_pred"] = merged_df["y_pred_x"] - merged_df["y_pred_y"]
            merged_df = merged_df[
                [
                    "model_x",
                    "pubchem_id",
                    "drug_name",
                    "cellosaurus_id",
                    "cell_line_name",
                    "y_true",
                    "y_pred",
                    "algorithm_x",
                    "rand_setting",
                    "test_mode",
                    "CV_split_x",
                ]
            ]
            self.df = merged_df.rename(
                columns={"model_x": "model", "algorithm_x": "algorithm", "CV_split_x": "CV_split"}
            )

    def draw_and_save(self, out_prefix: str | Path, out_suffix: str) -> None:
        """Draw regression plot and save as HTML.

        :param out_prefix: Output directory (for example ``results/my_run/regression_plots``).
        :param out_suffix: Filename suffix (for example ``LPO_drug_SimpleNeuralNetwork``).
        """
        self._draw()
        self.fig.write_html(Path(out_prefix) / f"regression_lines_{out_suffix}.html")

    def _draw(self):
        """Draw the regression plot."""
        print(f"Generating regression plots for {self.group_by}, normalize={self.normalize}, algorithm={self.model}...")
        self.df = self.df.groupby(self.group_by).filter(lambda x: len(x) > 1)
        pccs = self.df.groupby(self.group_by).apply(
            lambda x: pearsonr(x["y_true"], x["y_pred"])[0], include_groups=False
        )
        pccs = pccs.reset_index()
        pccs.columns = [self.group_by, "pcc"]
        self.df = self.df.merge(pccs, on=self.group_by)
        self._render_plot()

    @staticmethod
    def write_to_html(test_mode: str, f: TextIOWrapper, *_unused_args, **_kwargs) -> TextIOWrapper:
        """Insert regression plot links into the report HTML.

        :param test_mode: Evaluation test mode (for example ``"LPO"``).
        :param f: Open HTML file handle.
        :param _unused_args: Unused positional arguments.
        :param _kwargs: Keyword arguments; must include ``files``, a list of generated plot filenames.

        :returns: The same file handle after writing.
        """
        files: list[str] = _kwargs.get("files", [])
        f.write('<h2 id="regression_plots">Regression plots</h2>\n')
        f.write("<ul>\n")
        regr_files = [f for f in files if test_mode in f and f.startswith("regression_lines")]
        regr_files.sort()
        for regr_file in regr_files:
            f.write(f'<li><a href="regression_plots/{regr_file}" target="_blank">{regr_file}</a></li>\n')
        f.write("</ul>\n")
        return f

    def _render_plot(self):
        """Render the regression plot."""
        # sort df by group name
        df = self.df.sort_values(self.group_by)
        setting_title = self.model + " " + df["test_mode"].unique()[0]
        if self.normalize:
            setting_title += ", normalized by mean effects"
            hover_data = [
                "pcc",
                "cell_line_name",
                "cellosaurus_id",
                "drug_name",
                "pubchem_id",
                "algorithm",
            ]

        else:
            hover_data = ["pcc", "cell_line_name", "cellosaurus_id", "drug_name", "pubchem_id", "algorithm"]
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
        """Make a slider for the Pearson correlation coefficient.

        :param setting_title: Title of the plot.
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
