from typing import TextIO
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import operator
import math
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
import networkx

from drevalpy.evaluation import MINIMIZATION_METRICS
from drevalpy.visualization.outplot import OutPlot

matplotlib.use("agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "Avenir"


class CriticalDifferencePlot(OutPlot):
    def __init__(self, eval_results_preds: pd.DataFrame, metric="MSE"):
        eval_results_preds = eval_results_preds[
            ["algorithm", "CV_split", metric]
        ].rename(
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

    def draw_and_save(self, out_prefix: str, out_suffix: str) -> None:
        try:
            self.__draw__()
            path_out = f"{out_prefix}critical_difference_algorithms_{out_suffix}.svg"
            self.fig.savefig(path_out, bbox_inches="tight")
        except Exception as e:
            print(f"Error in drawing critical difference plot: {e}")

    def __draw__(self) -> None:
        self.fig = self.__draw_cd_diagram__(
            alpha=0.05, title=f"Critical Difference: {self.metric}", labels=True
        )

    @staticmethod
    def write_to_html(lpo_lco_ldo: str, f: TextIO, *args, **kwargs) -> TextIO:
        path_out_cd = f"critical_difference_plots/critical_difference_algorithms_{lpo_lco_ldo}.svg"
        f.write(f"<object data={path_out_cd}> </object>")
        return f

    def __draw_cd_diagram__(self, alpha=0.05, title=None, labels=False) -> plt.Figure:
        """
        Draws the critical difference diagram given the list of pairwise classifiers that are
        significant or not
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

        p_values, average_ranks, _ = wilcoxon_holm(
            alpha=alpha, df_perf=self.eval_results_preds
        )

        print(average_ranks)

        for p in p_values:
            print(p)

        graph_ranks(
            avranks=average_ranks.values,
            names=average_ranks.keys(),
            p_values=p_values,
            reverse=True,
            width=9,
            textspace=1.5,
            labels=labels,
            colors=plotly_colors,
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


# The code below is a modified version of the code available at https://github.com/hfawaz/cd-diagram
# Author: Hassan Ismail Fawaz <hassan.ismail-fawaz@uha.fr>
#         Germain Forestier <germain.forestier@uha.fr>
#         Jonathan Weber <jonathan.weber@uha.fr>
#         Lhassane Idoumghar <lhassane.idoumghar@uha.fr>
#         Pierre-Alain Muller <pierre-alain.muller@uha.fr>
# License: GPL3


# inspired from orange3 https://docs.orange.biolab.si/3/data-mining-library/reference/evaluation.cd.html
def graph_ranks(
    avranks,
    names,
    p_values,
    lowv=None,
    highv=None,
    width=6,
    textspace=1,
    reverse=False,
    labels=False,
    colors=None,
):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work.

    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
        labels (bool, optional): if set to `True`, the calculated avg rank
        values will be displayed
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

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

    def rankpos(rank):
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
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1.0 / height  # height factor
    wf = 1.0 / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color="k", **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=2, color="black")

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    tick = None
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
            cline - tick / 2 - 0.05,
            str(a),
            ha="center",
            va="bottom",
            size=16,
        )

    k = len(ssums)

    def filter_names(name):
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
    cliques = form_cliques(p_values, nnames)
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


def form_cliques(p_values, nnames):
    """
    This method forms the cliques
    """
    # first form the numpy matrix data
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)


def wilcoxon_holm(alpha=0.05, df_perf=None):
    """
    Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
    to reject the null's hypothesis
    """
    print(pd.unique(df_perf["classifier_name"]))
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame(
        {"count": df_perf.groupby(["classifier_name"]).size()}
    ).reset_index()
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts["count"].max()
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(
        df_counts.loc[df_counts["count"] == max_nb_datasets]["classifier_name"]
    )
    # test the null hypothesis using friedman before doing a post-hoc analysis
    friedman_p_value = friedmanchisquare(
        *(
            np.array(df_perf.loc[df_perf["classifier_name"] == c]["accuracy"])
            for c in classifiers
        )
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
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            # stop
            break
    # compute the average ranks to be returned (useful for drawing the cd diagram)
    # sort the dataframe of performances
    sorted_df_perf = df_perf.loc[
        df_perf["classifier_name"].isin(classifiers)
    ].sort_values(["classifier_name", "dataset_name"])
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
    average_ranks = (
        df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    )
    # return the p-values and the average ranks
    return p_values, average_ranks, max_nb_datasets
