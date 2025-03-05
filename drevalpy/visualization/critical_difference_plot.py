"""Draws the critical difference plot.

This method performs the following steps:

1. **Friedman Test**: First, it performs the Friedman test, which is a non-parametric statistical test used to detect
   differences in treatments across multiple test attempts. It compares the ranks of multiple groups and is
   suitable when there are repeated measurements for each group (as is the case here with cross-validation splits).
   The p-value of this test is used to assess whether there are any significant differences in the performance of the
   models.

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

import warnings
from io import TextIOWrapper

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
from scipy import stats

from ..evaluation import MINIMIZATION_METRICS
from ..pipeline_function import pipeline_function
from .outplot import OutPlot

matplotlib.use("agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "Avenir"
warnings.filterwarnings("ignore", category=FutureWarning, message=".*swapaxes.*")


class CriticalDifferencePlot(OutPlot):
    """
    Draws the critical difference diagram.

    Used by the pipeline!

    The critical difference diagram is used to compare the performance of multiple classifiers and show whether a
    model is significantly better than another model. This is calculated over the average ranks of the classifiers
    which is why there need to be at least 3 classifiers to draw the diagram. Because the ranks are calculated over
    the cross-validation splits and the significance threshold is set to 0.05, e.g., 10 CV folds are advisable.
    """

    @pipeline_function
    def __init__(self, eval_results_preds: pd.DataFrame, metric="MSE"):
        """
        Initializes the critical difference plot.

        :param eval_results_preds: evaluation results subsetted to predictions only (no randomizations etc)
        :param metric: to be used to assess the critical difference
        """
        eval_results_preds = eval_results_preds[["algorithm", "CV_split", metric]]
        if metric in MINIMIZATION_METRICS:
            eval_results_preds.loc[:, metric] = -eval_results_preds.loc[:, metric]

        self.eval_results_preds = eval_results_preds
        self.metric = metric

    @pipeline_function
    def draw_and_save(self, out_prefix: str, out_suffix: str) -> None:
        """
        Draws the critical difference plot and saves it to a file.

        :param out_prefix: e.g., results/my_run/critical_difference_plots/
        :param out_suffix: e.g., LPO
        """
        try:
            self._draw()
            path_out = f"{out_prefix}critical_difference_algorithms_{out_suffix}.svg"
            self.fig.savefig(path_out, bbox_inches="tight")
        except Exception as e:
            print(f"Error in drawing critical difference plot: {e}")

    def _draw(self) -> None:
        """Draws the critical difference plot."""
        friedman_p_value = stats.friedmanchisquare(
            *self.eval_results_preds.groupby("algorithm")[self.metric].apply(list)
        ).pvalue
        # transform: rows = CV_split, columns = algorithms, values = metric
        input_conover_friedman = self.eval_results_preds.pivot_table(
            index="CV_split", columns="algorithm", values=self.metric
        )
        self.test_results = sp.posthoc_conover_friedman(input_conover_friedman.to_numpy())
        average_ranks = input_conover_friedman.rank(ascending=False, axis=1).mean(axis=0)
        plt.title(
            f"Critical Difference Diagram: Metric: {self.metric}.\n"
            f"Overall Friedman-Chi2 p-value: {friedman_p_value:.2e}"
        )

        sp.critical_difference_diagram(ranks=average_ranks, sig_matrix=self.test_results)

        self.fig = plt.gcf()

    @staticmethod
    def write_to_html(lpo_lco_ldo: str, f: TextIOWrapper, *args, **kwargs) -> TextIOWrapper:
        """
        Inserts the critical difference plot into the HTML report file.

        :param lpo_lco_ldo: setting, e.g., LPO
        :param f: HTML report file
        :param args: not needed
        :param kwargs: not needed
        :returns: HTML report file
        """
        path_out_cd = f"critical_difference_plots/critical_difference_algorithms_{lpo_lco_ldo}.svg"
        f.write(f"<object data={path_out_cd}> </object>")
        return f
