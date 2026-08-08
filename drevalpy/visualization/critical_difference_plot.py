"""Draws the critical difference plot.

This method performs the following steps:

1. **Friedman Test**: First, it performs the Friedman test, which is a non-parametric statistical test used to detect
differences in treatments across multiple test attempts. It compares the ranks of multiple groups and is
suitable when there are repeated measurements for each group (as is the case here with cross-validation splits).
The p-value of this test is used to assess whether there are any significant differences in the performance of the
models. We use Benjamini/Hochberg correction for multiple testing.

2. **Post-hoc Conover Test**: If the Friedman test returns a significant result (p-value < 0.05), the post-hoc Conover
test can be used to identify pairs of algorithms that perform significantly different. This test is necessary
because the Friedman test only tells if there is a difference somewhere among the models, but not which ones are
different. The `scikit_posthocs` library is used for this step.

3. **Rank Calculation**: Next, the average ranks of each classifier across all cross-validation splits are computed.
The models are ranked based on their performance (lower ranks indicate better performance) and the average rank
across all splits is calculated for each model.

4. **Critical Difference Diagram**: Finally, the method draws the critical difference diagram. This diagram visually
displays the significant differences between the algorithms. A horizontal line groups a set of models that are
not significantly different. The critical difference is determined based on the post-hoc test results.
"""

import pathlib
import warnings
from io import TextIOWrapper
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.colors as pc
import scikit_posthocs as sp
from scipy import stats

from ..evaluation import MINIMIZATION_METRICS
from .critical_difference_layout import critical_difference_diagram as _critical_difference_diagram
from .outplot import OutPlot

matplotlib.use("agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "Helvetica Neue"
warnings.filterwarnings("ignore", category=FutureWarning, message=".*swapaxes.*")


class CriticalDifferencePlot(OutPlot):
    """Critical difference diagram comparing model ranks across CV splits.

    Requires at least three models; more CV folds improve pairwise significance
    testing at the default 0.05 threshold.
    """

    def __init__(self, eval_results_preds: pd.DataFrame, metric="MSE"):
        """Initialize critical difference plot.

        :param eval_results_preds: Evaluation results restricted to prediction runs.
        :param metric: Metric used for ranking (for example ``"MSE"``).

        :raises ValueError: If ``eval_results_preds`` is empty or lacks ``metric``.
        """
        eval_results_preds = eval_results_preds[["algorithm", "CV_split", metric]]
        if eval_results_preds.empty:
            raise ValueError(
                "Critical Difference Plot: The DataFrame is empty. Please provide a valid DataFrame with predictions."
            )
        if metric in MINIMIZATION_METRICS:
            eval_results_preds.loc[:, metric] = -eval_results_preds.loc[:, metric]

        self.eval_results_preds = eval_results_preds
        self.metric = metric
        self.fig: plt.Figure | None = None
        self.test_results: pd.DataFrame | None = None

    def draw_and_save(self, out_prefix: str | Path, out_suffix: str) -> None:
        """Draw critical difference plot and save SVG and HTML table.

        :param out_prefix: Output directory (for example ``results/my_run/critical_difference_plots``).
        :param out_suffix: Filename suffix (for example ``LPO``).

        :raises ValueError: If the figure or test results were not produced.
        """
        try:
            self._draw()
            path_out = Path(out_prefix) / f"critical_difference_algorithms_{out_suffix}.svg"
            if self.fig is None or self.test_results is None:
                raise ValueError("Figure is None. Cannot save the plot.")
            else:
                self.fig.savefig(path_out, bbox_inches="tight")
                plt.clf()
                self.test_results = self.test_results.round(4)
                self.test_results.to_html(Path(out_prefix) / f"critical_difference_algorithms_{out_suffix}.html")
        except Exception as e:
            print(f"Error in drawing critical difference plot: {e}")

    def _draw(self) -> None:
        """Draws the critical difference plot."""
        input_friedman = self.eval_results_preds.groupby("algorithm")[self.metric].apply(list)
        # check that all algorithms have the same number of CV splits, if not filter them out
        # table lengths of arrays:
        table_lengths = input_friedman.apply(len)
        # get the most common length
        most_common_length = table_lengths.mode().values[0]
        # filter out algorithms that do not have the most common length
        input_friedman = input_friedman[table_lengths == most_common_length]
        algorithms_included = set(input_friedman.index)
        friedman_p_value = stats.friedmanchisquare(*input_friedman).pvalue
        self.eval_results_preds = self.eval_results_preds[
            self.eval_results_preds["algorithm"].isin(algorithms_included)
        ]
        # transform: rows = CV_split, columns = algorithms, values = metric
        input_conover_friedman = self.eval_results_preds.pivot_table(
            index="CV_split", columns="algorithm", values=self.metric
        )
        self.test_results = pd.DataFrame(sp.posthoc_conover_friedman(input_conover_friedman, p_adjust="fdr_bh"))
        average_ranks = input_conover_friedman.rank(ascending=False, axis=1).mean(axis=0)
        plt.title(
            f"Critical Difference Diagram: Metric: {self.metric}.\n"
            f"Overall Friedman-Chi2 p-value: {friedman_p_value:.2e}",
            fontsize=20,
        )
        color_palette = dict()
        generated_colors = _generate_discrete_palette(len(input_conover_friedman.columns))
        for alg in input_conover_friedman.columns:
            color_palette[alg] = generated_colors.pop()

        _critical_difference_diagram(ranks=average_ranks, sig_matrix=self.test_results, color_palette=color_palette)

        self.fig = plt.gcf()

    @staticmethod
    def write_to_html(test_mode: str, f: TextIOWrapper, *_unused_args, **_unused_kwargs) -> TextIOWrapper:
        """Embed critical difference diagram and Conover table in the report HTML.

        :param test_mode: Evaluation test mode (for example ``"LPO"``).
        :param f: Open HTML file handle.
        :param _unused_args: Unused positional arguments.
        :param _unused_kwargs: Unused keyword arguments.

        :returns: The same file handle after writing.
        """
        path_out_cd = f"critical_difference_plots/critical_difference_algorithms_{test_mode}.svg"
        f.write(f"<object data={path_out_cd}> </object>")
        f.write(
            "<br><br>"
            "This diagram displays the mean rank of each model over all cross-validation splits: Within each CV "
            "split, the models are ranked according to their MSE. We calculate whether a model is significantly "
            "better than another one using the Friedman test and the post-hoc Conover test. "
            "The Friedman test shows whether there are overall differences between the models. After a significant"
            "Friedman test, the pairwise Conover test is performed to identify which models are significantly "
            "outperforming others. One line indicates which models are not significantly different from each "
            "other. The p-values are shown below. This can only be rendered if at least 3 models were run."
        )
        f.write("<br><br>")
        f.write("<h2>Results of the pairwise Post-Hoc Conover Test</h2>")
        f.write("<p>All p-values are adjusted with Benjamini-Hochberg correction.</p>")
        f.write("<br>")
        path_to_table = pathlib.Path(
            pathlib.Path(f.name).parent, f"critical_difference_plots/critical_difference_algorithms_{test_mode}.html"
        )
        if not path_to_table.exists():
            return f
        with open(path_to_table) as conover_results_f:
            conover_results = conover_results_f.readlines()
            conover_results[0] = conover_results[0].replace(
                '<table border="1" class="dataframe">',
                '<table class="display customDataTable" style="width:100%">',
            )
            for line in conover_results:
                f.write(line)
        return f


def _generate_discrete_palette(n_colors):
    # Get the base D3 categorical palette
    base_palette = pc.qualitative.D3
    base_n = len(base_palette)  # Number of available discrete colors

    if n_colors <= base_n:
        return base_palette[:n_colors]  # Use available colors directly

    # Convert HEX to RGB (0-1 range)
    base_rgb = np.array([matplotlib.colors.to_rgb(c) for c in base_palette])

    # Generate target indices in the interpolated space
    target_indices = np.linspace(0, base_n - 1, n_colors)

    # Interpolate in RGB space
    interpolated_rgb = np.array(
        [np.interp(target_indices, np.arange(base_n), base_rgb[:, i]) for i in range(3)]
    ).T  # Transpose to get (n_colors, 3)

    # Convert back to HEX
    interpolated_hex = [matplotlib.colors.to_hex(c) for c in interpolated_rgb]

    return interpolated_hex
