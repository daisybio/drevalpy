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

import pathlib
import warnings
from io import TextIOWrapper
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.colors as pc
import scikit_posthocs as sp
from matplotlib import pyplot
from matplotlib.axes import Axes
from pandas import DataFrame, Series
from scikit_posthocs import sign_array
from scipy import stats

from ..evaluation import MINIMIZATION_METRICS
from .outplot import OutPlot

matplotlib.use("agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "Helvetica Neue"
warnings.filterwarnings("ignore", category=FutureWarning, message=".*swapaxes.*")


class CriticalDifferencePlot(OutPlot):
    """
    Draws the critical difference diagram.

    The critical difference diagram is used to compare the performance of multiple classifiers and show whether a
    model is significantly better than another model. This is calculated over the average ranks of the classifiers
    which is why there need to be at least 3 classifiers to draw the diagram. Because the ranks are calculated over
    the cross-validation splits and the significance threshold is set to 0.05, e.g., 10 CV folds are advisable.
    """

    def __init__(self, eval_results_preds: pd.DataFrame, metric="MSE"):
        """
        Initializes the critical difference plot.

        :param eval_results_preds: evaluation results subsetted to predictions only (no randomizations etc)
        :param metric: to be used to assess the critical difference
        :raises ValueError: if eval_results_preds is empty or does not contain the metric
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
        self.fig: Optional[plt.Figure] = None
        self.test_results: Optional[pd.DataFrame] = None

    def draw_and_save(self, out_prefix: str, out_suffix: str) -> None:
        """
        Draws the critical difference plot and saves it to a file.

        :param out_prefix: e.g., results/my_run/critical_difference_plots/
        :param out_suffix: e.g., LPO
        :raises ValueError: if the figure is None or the test results are None
        """
        try:
            self._draw()
            path_out = f"{out_prefix}critical_difference_algorithms_{out_suffix}.svg"
            if self.fig is None or self.test_results is None:
                raise ValueError("Figure is None. Cannot save the plot.")
            else:
                self.fig.savefig(path_out, bbox_inches="tight")
                plt.clf()
                self.test_results = self.test_results.round(4)
                self.test_results.to_html(f"{out_prefix}critical_difference_algorithms_{out_suffix}.html")
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
        self.test_results = pd.DataFrame(sp.posthoc_conover_friedman(input_conover_friedman))
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
    def write_to_html(test_mode: str, f: TextIOWrapper, *args, **kwargs) -> TextIOWrapper:
        """
        Inserts the critical difference plot into the HTML report file.

        :param test_mode: test_mode, e.g., LPO
        :param f: HTML report file
        :param args: not needed
        :param kwargs: not needed
        :returns: HTML report file
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
        f.write("<h2>Results of Post-Hoc Conover Test</h2>")
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


def _critical_difference_diagram(
    ranks: Union[dict, Series],
    sig_matrix: DataFrame,
    *,
    color_palette: dict,
    ax: Optional[Axes] = None,
    label_fmt_left: str = "{label} ({rank:.2g})",
    label_fmt_right: str = "({rank:.2g}) {label}",
    label_props: Optional[dict] = None,
    marker_props: Optional[dict] = None,
    elbow_props: Optional[dict] = None,
    crossbar_props: Optional[dict] = None,
    text_h_margin: float = 0.01,
    left_only: bool = False,
) -> dict[str, list]:
    """
    Plot a Critical Difference diagram from ranks and post-hoc results.

    :param ranks : dict or Series
        Indicates the rank value for each sample or estimator (as keys or index).

    :param sig_matrix : DataFrame
        The corresponding p-value matrix outputted by post-hoc tests, with
        indices matching the labels in the ranks argument.

    :param ax : matplotlib.SubplotBase, optional
        The object in which the plot will be built. Gets the current Axes
        by default (if None is passed).

    :param label_fmt_left : str, optional
        The format string to apply to the labels on the left side. The keywords
        label and rank can be used to specify the sample/estimator name and
        rank value, respectively, by default '{label} ({rank:.2g})'.

    :param label_fmt_right : str, optional
        The same, but for the labels on the right side of the plot.
        By default '({rank:.2g}) {label}'.

    :param label_props : dict, optional
        Parameters to be passed to pyplot.text() when creating the labels,
        by default None.

    :param marker_props : dict, optional
        Parameters to be passed to pyplot.scatter() when plotting the rank
        markers on the axis, by default None.

    :param elbow_props : dict, optional
        Parameters to be passed to pyplot.plot() when creating the elbow lines,
        by default None.

    :param crossbar_props : dict, optional
        Parameters to be passed to pyplot.plot() when creating the crossbars
        that indicate lack of statistically significant difference. By default
        None.

    :param color_palette: dict
        Parameters to be passed when you need specific colors for each category

    :param text_h_margin : float, optional
        Space between the text labels and the nearest vertical line of an
        elbow, by default 0.01.

    :param left_only: boolean, optional
        Set all labels in a single left-sided block instead of splitting them
        into two block, one for the left and one for the right.
    :raises ValueError: If the color_palette keys are not consistent with the ranks.
    :returns: dict
    """
    # check color_palette consistency
    if isinstance(color_palette, dict) and ((len(set(ranks.keys()) & set(color_palette.keys()))) == len(ranks)):
        pass
    elif isinstance(color_palette, list) and (len(ranks) <= len(color_palette)):
        pass
    else:
        raise ValueError("color_palette keys are not consistent, or list size too small")

    elbow_props = elbow_props or {}
    marker_props = {"zorder": 3, **(marker_props or {})}
    label_props = {"va": "center", "fontsize": 16, "weight": "heavy", **(label_props or {})}
    crossbar_props = {
        "color": "k",
        "zorder": 3,
        "linewidth": 4,
        **(crossbar_props or {}),
    }

    ax = ax or pyplot.gca()
    ax.yaxis.set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.spines["top"].set_position("zero")

    # lists of artists to be returned
    markers = []
    elbows = []
    labels = []
    crossbars = []

    # True if pairwise comparison is NOT significant
    adj_matrix = DataFrame(
        1 - sign_array(sig_matrix),
        index=sig_matrix.index,
        columns=sig_matrix.columns,
        dtype=bool,
    )

    ranks = Series(ranks).sort_values()  # Standardize if ranks is dict
    if left_only:
        points_left = ranks
    else:
        left_points = len(ranks) // 2
        points_left, points_right = (
            ranks.iloc[:left_points],
            ranks.iloc[left_points:],
        )

    # for each algorithm: get the set of algorithms that are not significantly different
    crossbar_sets = dict()
    for alg, row in adj_matrix.iterrows():
        not_different = adj_matrix.columns[row].tolist()
        crossbar_sets[alg] = set(not_different).union({alg})

    # Create stacking of crossbars: make a crossbar of the fitting color for each algorithm
    crossbar_levels: list[list[set]] = []
    ypos = -0.5
    for alg in ranks.index:
        bar = crossbar_sets[alg]
        not_different = crossbar_sets[alg]
        if len(not_different) == 1:
            continue
        crossbar_levels.append([bar])

        crossbar_props["color"] = color_palette[alg]
        crossbars.append(
            ax.plot(
                # Adding a separate line between each pair enables showing a
                # marker over each elbow with crossbar_props={'marker': 'o'}.
                [ranks[i] for i in bar],
                [ypos] * len(bar),
                **crossbar_props,
            )
        )
        ypos -= 0.5

    lowest_crossbar_ypos = ypos

    def plot_items(points, xpos, label_fmt, color_palette, label_props):
        """
        Plot each marker + elbow + label.

        :param points: the points to plot
        :param xpos: the x position of the points
        :param label_fmt: the format of the label
        :param color_palette: the color palette to use
        :param label_props: the label properties
        """
        ypos = lowest_crossbar_ypos - 0.5
        for idx, (label, rank) in enumerate(points.items()):
            if not color_palette or len(color_palette) == 0:
                elbow, *_ = ax.plot(
                    [xpos, rank, rank],
                    [ypos, ypos, 0],
                    **elbow_props,
                )
            else:
                elbow, *_ = ax.plot(
                    [xpos, rank, rank],
                    [ypos, ypos, 0],
                    c=color_palette[label] if isinstance(color_palette, dict) else color_palette[idx],
                    **elbow_props,
                )

            elbows.append(elbow)
            curr_color = elbow.get_color()
            markers.append(ax.scatter(rank, 0, **{"color": curr_color, **marker_props}))
            labels.append(
                ax.text(
                    xpos,
                    ypos,
                    label_fmt.format(label=label, rank=rank),
                    color=curr_color,
                    **label_props,
                )
            )
            ypos -= 0.5

    plot_items(
        points_left,
        xpos=points_left.iloc[0] - text_h_margin,
        label_fmt=label_fmt_left,
        color_palette=color_palette,
        label_props={
            "ha": "right",
            **label_props,
        },
    )

    if not left_only:
        plot_items(
            points_right[::-1],
            xpos=points_right.iloc[-1] + text_h_margin,
            label_fmt=label_fmt_right,
            color_palette=color_palette,
            label_props={"ha": "left", **label_props},
        )

    return {
        "markers": markers,
        "elbows": elbows,
        "labels": labels,
        "crossbars": crossbars,
    }


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
