"""
Draw critical difference plot which shows whether a model is significantly better than another model.

Most code is a modified version of the code available at https://github.com/hfawaz/cd-diagram
Author: Hassan Ismail Fawaz <hassan.ismail-fawaz@uha.fr>, Germain Forestier <germain.forestier@uha.fr>,
Jonathan Weber <jonathan.weber@uha.fr>, Lhassane Idoumghar <lhassane.idoumghar@uha.fr>, Pierre-Alain Muller
<pierre-alain.muller@uha.fr>
License: GPL3
"""

import math
import operator
from io import TextIOWrapper
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import networkx
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from ..evaluation import MINIMIZATION_METRICS
from ..pipeline_function import pipeline_function
from .outplot import OutPlot

matplotlib.use("agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "Avenir"


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
        eval_results_preds = eval_results_preds[["algorithm", "CV_split", metric]].rename(
            columns={
                "algorithm": "classifier_name",
                "CV_split": "dataset_name",
                metric: "accuracy",
            }
        )
        if metric in MINIMIZATION_METRICS:
            eval_results_preds["accuracy"] = -eval_results_preds["accuracy"]

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
        self.fig = self._draw_cd_diagram(
            alpha=0.05,
            title=f"Critical Difference: {self.metric}",
            labels=True,
        )

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

    def _draw_cd_diagram(self, alpha=0.05, title=None, labels=False) -> plt.Figure:
        """
        Draws the critical difference diagram given the list of pairwise classifiers.

        :param alpha: significance level
        :param title: title of the plot
        :param labels: whether to display the average ranks
        :returns: the figure
        """
        # Standard Plotly colors
        plotly_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        p_values, average_ranks, _ = _wilcoxon_holm(df_perf=self.eval_results_preds, alpha=alpha)

        _graph_ranks(
            avranks=average_ranks.values.tolist(),
            names=list(average_ranks.keys()),
            p_values=p_values,
            colors=plotly_colors,
            reverse=True,
            width=9.0,
            textspace=1.5,
            labels=labels,
        )

        font = {
            "family": "sans-serif",
            "color": "black",
            "weight": "normal",
            "size": 22,
        }
        if title:
            plt.title(title, fontdict=font, y=0.9, x=0.5)
        return plt.gcf()


# inspired from orange3 https://docs.orange.biolab.si/3/data-mining-library/reference/evaluation.cd.html
def _graph_ranks(
    avranks: list[float],
    names: list[str],
    p_values: list[tuple[str, str, float, bool]],
    colors: list[str],
    lowv: int | None = None,
    highv: int | None = None,
    width: float = 9.0,
    textspace: float = 1.0,
    reverse: bool = False,
    labels: bool = False,
) -> None:
    """
    Draws a CD graph, which is used to display  the differences in methods' performance.

    See Janez Demsar, Statistical Comparisons of Classifiers over Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work. The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    :param avranks: list of float, average ranks of methods.
    :param names: list of str, names of methods.
    :param p_values: list of tuples, p-values of the methods.
    :param lowv: int, optional the lowest shown rank
    :param highv: int, optional, the highest shown rank
    :param width: int, optional, default width in inches (default: 6)
    :param textspace: int, optional, space on figure sides (in inches) for the method names (default: 1)
    :param reverse: bool, optional, if set to `True`, the lowest rank is on the right (default: `False`)
    :param labels: bool, optional, if set to `True`, the calculated avg rank values will be displayed
    :param colors: list of str, optional, list of colors for the methods
    """

    def nth(data: list[tuple[float, float]], position: int) -> list[float]:
        """
        Returns only nth element in a list.

        :param data: list (text_space, cline), (width - text_space, cline)
        :param position: position to return
        :returns: nth element in the list
        """
        position = lloc(data, position)
        return [a[position] for a in data]

    def lloc(data: list[tuple[float, float]], position: int) -> int:
        """
        List location in list of list structure.

        Enable the use of negative locations:
        -1 is the last element, -2 second last...

        :param data: list (text_space, cline), (width - text_space, cline)
        :param position: position to return
        :returns: location in the list
        """
        if position < 0:
            return len(data[0]) + position
        else:
            return position

    sums = avranks

    nnames = names
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank: float) -> float:
        """
        Calculate the position of the rank.

        :param rank: rank of the method
        :returns: textspace + scalewidth / (highv - lowv) * a
        """
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor("white")
    ax = fig.add_axes(rect=(0.0, 0.0, 1.0, 1.0))  # reverse y axis
    ax.set_axis_off()

    hf = 1.0 / height  # height factor
    wf = 1.0 / width

    def hfl(list_input):
        """
        List input multiplied by height factor.

        :param list_input: list of floats (cline)
        :returns: list of floats
        """
        return [a * hf for a in list_input]

    def wfl(list_input: list[float]) -> list[float]:
        """
        List input multiplied by width factor.

        :param list_input: list of floats (text_space)
        :returns: list of floats
        """
        return [a * wf for a in list_input]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(list_input: list[tuple[float, float]], color: str = "k", **kwargs) -> None:
        """
        Input is a list of pairs of points.

        :param list_input: (text_space, cline), (width - text_space, cline)
        :param color: color of the line
        :param kwargs: additional arguments for plotting
        """
        ax.plot(wfl(nth(list_input, 0)), hfl(nth(list_input, 1)), color=color, **kwargs)

    def text(x: float, y: float, s: str, *args, **kwargs):
        """
        Add text to the plot.

        :param x: x position
        :param y: y position
        :param s: text to display
        :param args: additional arguments
        :param kwargs: additional keyword arguments
        """
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line(
        [(textspace, cline), (width - textspace, cline)],
        linewidth=2,
        color="black",
    )

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line(
            [(rankpos(a), cline - tick / 2), (rankpos(a), cline)],
            linewidth=2,
            color="black",
        )

    for a in range(lowv, highv + 1):
        text(
            rankpos(a),
            cline - bigtick / 2 - 0.05,
            str(a),
            ha="center",
            va="bottom",
            size=16,
        )

    k = len(ssums)

    def filter_names(name: str) -> str:
        """
        Filter the names.

        :param name: name of the method
        :returns: name of the method
        """
        return name

    space_between_names = 0.24

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line(
            [
                (rankpos(ssums[i]), cline),
                (rankpos(ssums[i]), chei),
                (textspace - 0.1, chei),
            ],
            linewidth=linewidth,
            color=colors[0],
        )
        if labels:
            text(
                textspace + 0.3,
                chei - 0.075,
                format(ssums[i], ".4f"),
                ha="right",
                va="center",
                size=10,
            )
        text(
            textspace - 0.2,
            chei,
            filter_names(nnames[i]),
            ha="right",
            va="center",
            size=16,
        )

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line(
            [
                (rankpos(ssums[i]), cline),
                (rankpos(ssums[i]), chei),
                (textspace + scalewidth + 0.1, chei),
            ],
            linewidth=linewidth,
            color=colors[0],
        )
        if labels:
            text(
                textspace + scalewidth - 0.3,
                chei - 0.075,
                format(ssums[i], ".4f"),
                ha="left",
                va="center",
                size=10,
            )
        text(
            textspace + scalewidth + 0.2,
            chei,
            filter_names(nnames[i]),
            ha="left",
            va="center",
            size=16,
        )

    start = cline + 0.2
    side = -0.02
    height = 0.1

    # draw no significant lines
    # get the cliques
    cliques = _form_cliques(p_values, nnames)
    i = 1
    achieved_half = False
    print(nnames)
    for clq in cliques:
        if len(clq) == 1:
            continue
        print(clq)
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and (not achieved_half):
            start = cline + 0.25
            achieved_half = True
        line(
            [
                (rankpos(ssums[min_idx]) - side, start),
                (rankpos(ssums[max_idx]) + side, start),
            ],
            linewidth=linewidth_sign,
            color=colors[2],
        )
        start += height


def _form_cliques(p_values: list[tuple[str, str, float, bool]], nnames: list[str]) -> Any:
    """
    This method forms the cliques.

    :param p_values: list of tuples, p-values of the methods strucutred as (Method1, Method2, p-value, is_significant)
    :param nnames: list of str, names of the methods
    :returns: cliques
    """
    # first form the numpy matrix data
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] is False:
            i = int(np.where(np.array(nnames) == p[0])[0][0])
            j = int(np.where(np.array(nnames) == p[1])[0][0])
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)


def _wilcoxon_holm(
    df_perf: pd.DataFrame, alpha: float = 0.05
) -> tuple[list[tuple[str, str, float, bool]], pd.Series, int]:
    """
    Applies the Wilcoxon signed rank test between algorithm pair and then use Holm to reject the null hypothesis.

    Returns the p-values in a format of (Method1, Method2, p-value, is_significant), the average ranks in a format of
    pd.Series(Method: avg_rank), and the maximum number of datasets tested (=n_cv_folds).

    :param alpha: significance level
    :param df_perf: the dataframe containing the performance of the algorithms
    :returns: the p-values, the average ranks, and the maximum number of datasets tested
    """
    print(pd.unique(df_perf["classifier_name"]))
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({"count": df_perf.groupby(["classifier_name"]).size()}).reset_index()
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts["count"].max()
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(df_counts.loc[df_counts["count"] == max_nb_datasets]["classifier_name"])
    # test the null hypothesis using friedman before doing a post-hoc analysis
    friedman_p_value = friedmanchisquare(
        *(np.array(df_perf.loc[df_perf["classifier_name"] == c]["accuracy"]) for c in classifiers)
    )[1]
    if friedman_p_value >= alpha:
        # then the null hypothesis over the entire classifiers cannot be rejected
        print("the null hypothesis over the entire classifiers cannot be rejected")
        exit()
    # get the number of classifiers
    m = len(classifiers)
    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    p_values = []
    # loop through the algorithms to compare pairwise
    for i in range(m - 1):
        # get the name of classifier one
        classifier_1 = classifiers[i]
        # get the performance of classifier one
        perf_1 = np.array(
            df_perf.loc[df_perf["classifier_name"] == classifier_1]["accuracy"],
            dtype=np.float64,
        )
        for j in range(i + 1, m):
            # get the name of the second classifier
            classifier_2 = classifiers[j]
            # get the performance of classifier one
            perf_2 = np.array(
                df_perf.loc[df_perf["classifier_name"] == classifier_2]["accuracy"],
                dtype=np.float64,
            )
            # calculate the p_value
            p_value = wilcoxon(perf_1, perf_2, zero_method="pratt")[1]
            # append to the list
            p_values.append((classifier_1, classifier_2, p_value, False))
    # get the number of hypothesis
    k = len(p_values)
    # sort the list in acsending manner of p-value
    p_values.sort(key=operator.itemgetter(2))

    # loop through the hypothesis
    for i in range(k):
        # correct alpha with holm
        new_alpha = float(alpha / (k - i))
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (
                p_values[i][0],
                p_values[i][1],
                p_values[i][2],
                True,
            )
        else:
            # stop
            break
    # compute the average ranks to be returned (useful for drawing the cd diagram)
    # sort the dataframe of performances
    sorted_df_perf = df_perf.loc[df_perf["classifier_name"].isin(classifiers)].sort_values(
        ["classifier_name", "dataset_name"]
    )
    # get the rank data
    rank_data = np.array(sorted_df_perf["accuracy"]).reshape(m, max_nb_datasets)

    # create the data frame containg the accuracies
    df_ranks = pd.DataFrame(
        data=rank_data,
        index=np.sort(classifiers),
        columns=np.unique(sorted_df_perf["dataset_name"]),
    )

    # number of wins
    dfff = df_ranks.rank(ascending=False)
    print(dfff[dfff == 1.0].sum(axis=1))

    # average the ranks
    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    # return the p-values and the average ranks
    return p_values, average_ranks, max_nb_datasets
