"""Parent class for Violin and Heatmap plots of performance measures over CV runs."""

from io import TextIOWrapper

import pandas as pd

from drevalpy.visualization.outplot import OutPlot


class VioHeat(OutPlot):
    """Parent class for Violin and Heatmap plots of performance measures over CV runs."""

    def __init__(self, df: pd.DataFrame, normalized_metrics=False, whole_name=False):
        """
        Initialize the VioHeat class.

        :param df: evaluation results, either overall or per algorithm
        :param normalized_metrics: whether the metrics are normalized
        :param whole_name: whether the whole name should be displayed
        """
        self.df = df.sort_index()
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
        self.normalized_metrics = normalized_metrics
        self.whole_name = whole_name
        if self.normalized_metrics:
            self.all_metrics = [metric for metric in self.all_metrics if "normalized" in metric]
        else:
            self.all_metrics = [metric for metric in self.all_metrics if "normalized" not in metric]

    def draw_and_save(self, out_prefix: str, out_suffix: str) -> None:
        """
        Draw and save the plot.

        :param out_prefix: e.g., results/my_run/heatmaps/
        :param out_suffix: e.g., algorithms_normalized
        """
        pass

    def _draw(self) -> None:
        pass

    @staticmethod
    def write_to_html(lpo_lco_ldo: str, f: TextIOWrapper, *args, **kwargs) -> TextIOWrapper:
        """
        Write the Violin and Heatmap plots into the result HTML file.

        :param lpo_lco_ldo: setting, e.g., LPO
        :param f: result HTML file
        :param args: additional arguments
        :param kwargs: additional keyword arguments, in this case, the plot type and the files
        :returns: the result HTML file
        """
        plot: str = kwargs.get("plot", "")
        files: list[str] = kwargs.get("files", [])

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
                lpo_lco_ldo in f
                and f.startswith(prefix)
                and f != f"{prefix}_{lpo_lco_ldo}.html"
                and f != f"{prefix}_{lpo_lco_ldo}_normalized.html"
            )
        ]
        f.write(f"<h2 id={nav_id!r}>{plot} Plots of Performance Measures over CV runs</h2>\n")
        f.write(f"<h3>{plot} plots comparing all models</h3>\n")
        f.write(
            f'<iframe src="{dir_name}/{prefix}_algorithms_{lpo_lco_ldo}.html" width="100%" height="100%" '
            f'frameBorder="0"></iframe>\n'
        )
        f.write(f"<h3>{plot} plots comparing all models with normalized metrics</h3>\n")
        f.write(
            "Before calculating the evaluation metrics, all values were normalized by the mean of the drug or cell "
            "line. Since this only influences the R^2 and the correlation metrics, the error metrics are not shown. \n"
        )
        f.write(
            f'<iframe src="{dir_name}/{prefix}_algorithms_{lpo_lco_ldo}_normalized.html" width="100%" height="100%" '
            f'frameBorder="0"></iframe>\n'
        )
        f.write(f"<h3>{plot} plots comparing performance measures for tests within each model</h3>\n")
        f.write("<ul>")
        for plot in plot_list:
            f.write(f'<li><a href="{dir_name}/{plot}" target="_blank">{plot}</a></li>\n')
        f.write("</ul>\n")
        return f
