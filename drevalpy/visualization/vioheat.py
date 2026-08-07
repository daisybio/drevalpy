"""Parent class for Violin and Heatmap plots of performance measures over CV runs."""

from io import TextIOWrapper
from pathlib import Path

import pandas as pd

from drevalpy.visualization.outplot import OutPlot


class VioHeat(OutPlot):
    """Parent class for violin and heatmap plots over CV runs."""

    def __init__(self, df: pd.DataFrame, normalized_metrics=False, whole_name=False):
        """Initialize shared violin/heatmap state.

        :param df: Evaluation results (overall or per algorithm).
        :param normalized_metrics: Whether to show only normalized metric columns.
        :param whole_name: Whether to display full algorithm setting labels.
        """
        self.df = df.sort_index()
        self.all_metrics = [
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
