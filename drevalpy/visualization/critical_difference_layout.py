"""Layout logic for critical-difference diagrams."""

from __future__ import annotations

from typing import Optional, Union

from matplotlib import pyplot
from matplotlib.axes import Axes
from pandas import DataFrame, Series
from scikit_posthocs import sign_array


def _validate_color_palette(ranks: Series, color_palette: dict | list) -> None:
    if isinstance(color_palette, dict) and len(set(ranks.keys()) & set(color_palette.keys())) == len(ranks):
        return
    if isinstance(color_palette, list) and len(ranks) <= len(color_palette):
        return
    raise ValueError("color_palette keys are not consistent, or list size too small")


def _split_rank_points(ranks: Series, left_only: bool) -> tuple[Series, Optional[Series]]:
    ranks = Series(ranks).sort_values()
    if left_only:
        return ranks, None
    left_points = len(ranks) // 2
    return ranks.iloc[:left_points], ranks.iloc[left_points:]


def _nonsignificant_adjacency(sig_matrix: DataFrame) -> DataFrame:
    return DataFrame(
        1 - sign_array(sig_matrix),
        index=sig_matrix.index,
        columns=sig_matrix.columns,
        dtype=bool,
    )


def _crossbar_sets_from_adjacency(adj_matrix: DataFrame) -> dict[str, set[str]]:
    crossbar_sets: dict[str, set[str]] = {}
    for alg, row in adj_matrix.iterrows():
        not_different = adj_matrix.columns[row].tolist()
        crossbar_sets[alg] = set(not_different).union({alg})
    return crossbar_sets


def _draw_crossbars(
    ax: Axes,
    ranks: Series,
    crossbar_sets: dict[str, set[str]],
    color_palette: dict,
    crossbar_props: dict,
) -> tuple[list, float]:
    crossbars: list = []
    ypos = -0.5
    for alg in ranks.index:
        bar = crossbar_sets[alg]
        if len(bar) == 1:
            continue
        props = {**crossbar_props, "color": color_palette[alg]}
        crossbars.append(ax.plot([ranks[i] for i in bar], [ypos] * len(bar), **props))
        ypos -= 0.5
    return crossbars, ypos


def _plot_rank_items(
    ax: Axes,
    points: Series,
    *,
    xpos: float,
    label_fmt: str,
    color_palette: dict | list,
    label_props: dict,
    elbow_props: dict,
    marker_props: dict,
    ypos_start: float,
) -> tuple[list, list, list]:
    markers: list = []
    elbows: list = []
    labels: list = []
    ypos = ypos_start
    for idx, (label, rank) in enumerate(points.items()):
        color = None
        if color_palette:
            color = color_palette[label] if isinstance(color_palette, dict) else color_palette[idx]
        plot_kwargs = {**elbow_props}
        if color is not None:
            plot_kwargs["c"] = color
        elbow, *_ = ax.plot([xpos, rank, rank], [ypos, ypos, 0], **plot_kwargs)
        elbows.append(elbow)
        curr_color = elbow.get_color()
        markers.append(ax.scatter(rank, 0, **{"color": curr_color, **marker_props}))
        labels.append(ax.text(xpos, ypos, label_fmt.format(label=label, rank=rank), color=curr_color, **label_props))
        ypos -= 0.5
    return markers, elbows, labels


def critical_difference_diagram(
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
    """Plot a critical difference diagram from ranks and post-hoc results.

    :param ranks: Average ranks per algorithm (dict or Series).
    :param sig_matrix: Pairwise significance matrix from a post-hoc test.
    :param color_palette: Map from algorithm name to color.
    :param ax: Optional matplotlib axes; defaults to the current axes.
    :param label_fmt_left: Format string for left-side rank labels.
    :param label_fmt_right: Format string for right-side rank labels.
    :param label_props: Extra matplotlib text properties.
    :param marker_props: Extra matplotlib marker properties.
    :param elbow_props: Extra matplotlib line properties for elbows.
    :param crossbar_props: Extra matplotlib line properties for crossbars.
    :param text_h_margin: Horizontal margin for label placement.
    :param left_only: If ``True``, draw ranks only on the left side.

    :returns: Dict with drawn matplotlib artists grouped by type.
    """
    _validate_color_palette(Series(ranks), color_palette)

    elbow_props = elbow_props or {}
    marker_props = {"zorder": 3, **(marker_props or {})}
    label_props = {"va": "center", "fontsize": 16, "weight": "heavy", **(label_props or {})}
    crossbar_props = {"color": "k", "zorder": 3, "linewidth": 4, **(crossbar_props or {})}

    ax = ax or pyplot.gca()
    ax.yaxis.set_visible(False)
    for spine in ("right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.spines["top"].set_position("zero")

    adj_matrix = _nonsignificant_adjacency(sig_matrix)
    points_left, points_right = _split_rank_points(ranks, left_only)
    crossbar_sets = _crossbar_sets_from_adjacency(adj_matrix)
    crossbars, lowest_y = _draw_crossbars(ax, Series(ranks).sort_values(), crossbar_sets, color_palette, crossbar_props)

    markers, elbows, labels = _plot_rank_items(
        ax,
        points_left,
        xpos=points_left.iloc[0] - text_h_margin,
        label_fmt=label_fmt_left,
        color_palette=color_palette,
        label_props={"ha": "right", **label_props},
        elbow_props=elbow_props,
        marker_props=marker_props,
        ypos_start=lowest_y - 0.5,
    )

    if points_right is not None:
        m2, e2, l2 = _plot_rank_items(
            ax,
            points_right[::-1],
            xpos=points_right.iloc[-1] + text_h_margin,
            label_fmt=label_fmt_right,
            color_palette=color_palette,
            label_props={"ha": "left", **label_props},
            elbow_props=elbow_props,
            marker_props=marker_props,
            ypos_start=lowest_y - 0.5,
        )
        markers.extend(m2)
        elbows.extend(e2)
        labels.extend(l2)

    return {"markers": markers, "elbows": elbows, "labels": labels, "crossbars": crossbars}
