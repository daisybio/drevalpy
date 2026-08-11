"""Parent class for Violin and Heatmap plots of performance measures over CV runs."""

from __future__ import annotations

from io import TextIOWrapper
from typing import TYPE_CHECKING

import pandas as pd
from upath import UPath as Path

from drevalpy.visualization.outplot import OutPlot
from drevalpy.visualization.plot_requirements import PlotRequirement

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult

_ALL_METRICS = [
    "R^2",
    "R^2: normalized",
    "Pearson",
    "Pearson: normalized",
    "Spearman",
    "Spearman: normalized",
    "Kendall",
    "Kendall: normalized",
    "MSE",
    "RMSE",
    "MAE",
]


def _build_df_from_experiment(result: ExperimentResult) -> pd.DataFrame:
    """Build a flat DataFrame from an ExperimentResult matching the legacy column layout.

    Columns: algorithm, rand_setting, test_mode, CV_split, and one column per metric.
    """
    rows: list[dict] = []
    for model in result.models:
        for run in model.runs:
            row: dict = {
                "algorithm": run.model_name,
                "rand_setting": (
                    f"{run.randomization[0]}_{run.randomization[1]}" if run.randomization else "predictions"
                ),
                "test_mode": result.split_mode,
                "CV_split": run.fold_index,
            }
            row.update(run.metrics)
            rows.append(row)
    return pd.DataFrame(rows)


class VioHeat(OutPlot):
    """Parent class for violin and heatmap plots over CV runs."""

    result_type: str = "ExperimentResult"
    requirements: frozenset = frozenset({PlotRequirement.MULTIPLE_FOLDS})

    def __init__(
        self,
        result: ExperimentResult | None = None,
        *,
        df: pd.DataFrame | None = None,
        normalized_metrics: bool = False,
        whole_name: bool = False,
    ):
        """Initialize shared violin/heatmap state.

        :param result: Typed experiment result (preferred path).
        :param df: Legacy evaluation results DataFrame.
        :param normalized_metrics: Whether to show only normalized metric columns.
        :param whole_name: Whether to display full algorithm setting labels.
        """
        if result is not None:
            self.df = _build_df_from_experiment(result).sort_index()
        elif df is not None:
            self.df = df.sort_index()
        else:
            raise ValueError("Either 'result' or 'df' must be provided")

        self.all_metrics = list(_ALL_METRICS)
        self.normalized_metrics = normalized_metrics
        self.whole_name = whole_name
        if self.normalized_metrics:
            self.all_metrics = [metric for metric in self.all_metrics if "normalized" in metric]
        else:
            self.all_metrics = [metric for metric in self.all_metrics if "normalized" not in metric]

    def draw_and_save(self, out_prefix: str | Path, out_suffix: str) -> None:
        """Draw and save the plot (implemented by subclasses).

        :param out_prefix: Output directory path.
        :param out_suffix: Filename suffix for the saved artifact.
        """
        pass

    def _draw(self) -> None:
        pass

    @staticmethod
    def write_to_html(test_mode: str, f: TextIOWrapper, *_unused_args, **_kwargs) -> TextIOWrapper:
        """Write violin or heatmap sections into the report HTML.

        :param test_mode: Evaluation test mode (for example ``"LPO"``).
        :param f: Open HTML file handle.
        :param _unused_args: Unused positional arguments.
        :param _kwargs: Keyword arguments with ``plot`` (``Violin`` or ``Heatmap``) and ``files`` list.

        :returns: The same file handle after writing.
        """
        plot: str = _kwargs.get("plot", "")
        files: list[str] = _kwargs.get("files", [])

        if plot == "Violin":
            nav_id = "violin"
            dir_name = "violin_plots"
            prefix = "violin"
        else:
            nav_id = "heatmap"
            dir_name = "heatmaps"
            prefix = "heatmap"
        plot_list = [
            f
            for f in files
            if (
                test_mode in f
                and f.startswith(prefix)
                and f != f"{prefix}_algorithms_{test_mode}.html"
                and f != f"{prefix}_algorithms_{test_mode}_normalized.html"
            )
        ]
        f.write(f"<h2 id={nav_id!r}>{plot} Plots of Performance Measures over CV runs</h2>\n")
        f.write(f"<h3>{plot} plots comparing all models</h3>\n")
        if plot == "Violin":
            f.write(
                "To focus on a specific metric, choose it in the dropdown menu in the top right corner."
                "You can investigate the distribution of the performance measures by hovering over the plot.\n"
                "To select/exclude specific algorithms, (double-)click them in the legend."
            )
        elif plot == "Heatmap":
            f.write(
                "Unnormalized metrics collapsed over all CV runs with mean and standard deviation.\n"
                "The strictly standardized mean difference is a measure of effect size which is calculated "
                "pairwise. For two models, it is calculated as [mean1 - mean2] / [sqrt(var1 + var2)] for a "
                "specific measure. The larger the absolute SSMD, the stronger the effect (a strong effect could, "
                "is e.g., a |SSMD| > 2 ).\n"
            )
        f.write(
            f'<iframe src="{dir_name}/{prefix}_algorithms_{test_mode}.html" width="100%" height="100%" '
            f'frameBorder="0"></iframe>\n'
        )
        f.write(f"<h3>{plot} plots comparing all models with normalized metrics</h3>\n")
        f.write(
            "Before calculating the evaluation metrics, all values were normalized by the predictions of the "
            "NaiveMeanEffectsPredictor. Since this only influences the R^2 and the correlation metrics, the error "
            "metrics are not shown. \n"
        )
        f.write(
            f'<iframe src="{dir_name}/{prefix}_algorithms_{test_mode}_normalized.html" width="100%" height="100%" '
            f'frameBorder="0"></iframe>\n'
        )
        f.write(f"<h3>{plot} plots comparing performance measures for tests within each model</h3>\n")
        f.write("<ul>")
        for plot in plot_list:
            f.write(f'<li><a href="{dir_name}/{plot}" target="_blank">{plot}</a></li>\n')
        f.write("</ul>\n")
        return f
