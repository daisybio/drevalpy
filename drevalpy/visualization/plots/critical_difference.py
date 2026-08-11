"""Critical difference diagram visualization (Matplotlib via ImageVisualization)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.colors as pc
import scikit_posthocs as sp
from matplotlib.axes import Axes
from scikit_posthocs import sign_array
from scipy import stats

from drevalpy.evaluation import MINIMIZATION_METRICS
from drevalpy.visualization.base import ImageVisualization
from drevalpy.visualization.registry import visualization_registry
from drevalpy.visualization.requirements import PlotRequirement

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult

matplotlib.use("agg")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*swapaxes.*")


def _build_cd_df(result: ExperimentResult, metric: str) -> pd.DataFrame:
    """Build DataFrame for CD plot: columns algorithm, CV_split, <metric>."""
    rows: list[dict] = []
    for model in result.models:
        for run in model.runs:
            if run.randomization is not None:
                continue
            rows.append(
                {
                    "algorithm": run.model_name,
                    "CV_split": run.fold_index,
                    metric: run.metrics.get(metric, float("nan")),
                }
            )
    return pd.DataFrame(rows)


def _generate_discrete_palette(n_colors: int) -> list[str]:
    base_palette = pc.qualitative.D3
    base_n = len(base_palette)
    if n_colors <= base_n:
        return list(base_palette[:n_colors])
    base_rgb = np.array([matplotlib.colors.to_rgb(c) for c in base_palette])
    target_indices = np.linspace(0, base_n - 1, n_colors)
    interpolated_rgb = np.array([np.interp(target_indices, np.arange(base_n), base_rgb[:, i]) for i in range(3)]).T
    return [matplotlib.colors.to_hex(c) for c in interpolated_rgb]


# --- CD layout logic (self-contained from legacy) ---


def _nonsignificant_adjacency(sig_matrix: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        1 - sign_array(sig_matrix),
        index=sig_matrix.index,
        columns=sig_matrix.columns,
        dtype=bool,
    )


def _crossbar_sets_from_adjacency(adj_matrix: pd.DataFrame) -> dict[str, set[str]]:
    crossbar_sets: dict[str, set[str]] = {}
    for alg, row in adj_matrix.iterrows():
        not_different = adj_matrix.columns[row].tolist()
        crossbar_sets[alg] = set(not_different).union({alg})
    return crossbar_sets


def _draw_crossbars(
    ax: Axes,
    ranks: pd.Series,
    crossbar_sets: dict[str, set[str]],
    color_palette: dict,
    crossbar_props: dict,
) -> float:
    ypos = -0.5
    for alg in ranks.index:
        bar = crossbar_sets[alg]
        if len(bar) == 1:
            continue
        props = {**crossbar_props, "color": color_palette[alg]}
        ax.plot([ranks[i] for i in bar], [ypos] * len(bar), **props)
        ypos -= 0.5
    return ypos


def _plot_rank_items(
    ax: Axes,
    points: pd.Series,
    *,
    xpos: float,
    label_fmt: str,
    color_palette: dict,
    label_props: dict,
    elbow_props: dict,
    marker_props: dict,
    ypos_start: float,
) -> None:
    ypos = ypos_start
    for label, rank in points.items():
        color = color_palette[label]
        plot_kwargs = {**elbow_props, "c": color}
        ax.plot([xpos, rank, rank], [ypos, ypos, 0], **plot_kwargs)
        ax.scatter(rank, 0, color=color, **marker_props)
        ax.text(xpos, ypos, label_fmt.format(label=label, rank=rank), color=color, **label_props)
        ypos -= 0.5


def _critical_difference_diagram(
    ranks: pd.Series,
    sig_matrix: pd.DataFrame,
    color_palette: dict,
    ax: Axes | None = None,
) -> None:
    """Draw the critical difference diagram on the given axes."""
    elbow_props: dict = {}
    marker_props = {"zorder": 3}
    label_props = {"va": "center", "fontsize": 16, "weight": "heavy"}
    crossbar_props = {"color": "k", "zorder": 3, "linewidth": 4}
    text_h_margin = 0.01

    ax = ax or plt.gca()
    ax.yaxis.set_visible(False)
    for spine in ("right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.spines["top"].set_position("zero")

    adj_matrix = _nonsignificant_adjacency(sig_matrix)
    ranks_sorted = ranks.sort_values()
    crossbar_sets = _crossbar_sets_from_adjacency(adj_matrix)
    lowest_y = _draw_crossbars(ax, ranks_sorted, crossbar_sets, color_palette, crossbar_props)

    left_points_n = len(ranks_sorted) // 2
    points_left = ranks_sorted.iloc[:left_points_n]
    points_right = ranks_sorted.iloc[left_points_n:]

    _plot_rank_items(
        ax,
        points_left,
        xpos=points_left.iloc[0] - text_h_margin,
        label_fmt="{label} ({rank:.2g})",
        color_palette=color_palette,
        label_props={"ha": "right", **label_props},
        elbow_props=elbow_props,
        marker_props=marker_props,
        ypos_start=lowest_y - 0.5,
    )

    if len(points_right) > 0:
        _plot_rank_items(
            ax,
            points_right[::-1],
            xpos=points_right.iloc[-1] + text_h_margin,
            label_fmt="({rank:.2g}) {label}",
            color_palette=color_palette,
            label_props={"ha": "left", **label_props},
            elbow_props=elbow_props,
            marker_props=marker_props,
            ypos_start=lowest_y - 0.5,
        )


@visualization_registry.register(
    "critical_difference",
    "Critical difference diagram with Friedman test and model rankings",
    requirements=frozenset({PlotRequirement.MULTIPLE_MODELS, PlotRequirement.MULTIPLE_FOLDS}),
)
class CriticalDifferenceVisualization(ImageVisualization):
    """Critical difference rank diagram using Matplotlib."""

    def __init__(self) -> None:
        """Initialize with empty state."""
        self._result: ExperimentResult | None = None
        self._metric: str = "MSE"
        self._fig: plt.Figure | None = None

    def compute(self, result: ExperimentResult, metric: str = "MSE") -> None:
        """Compute rankings and Friedman test, then create the CD figure.

        :param result: Experiment result with multiple models and folds.
        :param metric: Metric to rank models by.
        """
        self._result = result
        self._metric = metric
        self._fig = self._create_figure()

    def _create_figure(self) -> plt.Figure:
        """Create the critical difference diagram figure."""
        result = self._result
        metric = self._metric

        eval_df = _build_cd_df(result, metric)
        if eval_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return fig

        if metric in MINIMIZATION_METRICS:
            eval_df[metric] = -eval_df[metric]

        input_friedman = eval_df.groupby("algorithm")[metric].apply(list)
        table_lengths = input_friedman.apply(len)
        most_common_length = table_lengths.mode().values[0]
        input_friedman = input_friedman[table_lengths == most_common_length]
        algorithms_included = set(input_friedman.index)

        friedman_p_value = stats.friedmanchisquare(*input_friedman).pvalue

        eval_df = eval_df[eval_df["algorithm"].isin(algorithms_included)]
        input_conover = eval_df.pivot_table(index="CV_split", columns="algorithm", values=metric)
        test_results = pd.DataFrame(sp.posthoc_conover_friedman(input_conover, p_adjust="fdr_bh"))
        average_ranks = input_conover.rank(ascending=False, axis=1).mean(axis=0)

        fig, ax = plt.subplots(figsize=(12, max(4, len(algorithms_included) * 0.8)))
        plt.sca(ax)
        plt.title(
            f"Critical Difference Diagram: Metric: {metric}.\nOverall Friedman-Chi2 p-value: {friedman_p_value:.2e}",
            fontsize=20,
        )

        generated_colors = _generate_discrete_palette(len(input_conover.columns))
        color_palette = {alg: generated_colors[i] for i, alg in enumerate(input_conover.columns)}

        _critical_difference_diagram(ranks=average_ranks, sig_matrix=test_results, color_palette=color_palette, ax=ax)

        return fig
