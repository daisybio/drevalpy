"""Drawing helpers of the leaderboard: theme, ranked bar panels, title, subtitle and legend.

``matplotlib`` is imported inside the drawing helpers rather than at module scope:
``drevalpy.registry`` imports every builtin visualization on ``import drevalpy``, so a
module-scope import would put the whole pyplot stack on the startup path of every CLI
invocation. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyBboxPatch

# --- Theme ---
DARK_THEME = {
    "background": "#0d1117",
    "surface": "#2d2d2d",
    "text": "#ece7e4",
    "text_secondary": "#a0a0a0",
    "grid": "#30363d",
}

COMPETITOR_COLOR = "#6A5ACD"


def get_bar_color(rank: int, is_baseline: bool) -> dict[str, Any]:
    """Return the bar color and opacity: grey for baselines, medals for the top three, purple otherwise."""
    if is_baseline:
        return {"color": "#5a5a5a", "alpha": 1.0}
    medal_colors = ["#F4D03F", "#BDC3C7", "#E67E22"]
    if rank < len(medal_colors):
        return {"color": medal_colors[rank], "alpha": 1.0}
    return {"color": COMPETITOR_COLOR, "alpha": 0.85}


def draw_bar(
    ax: Axes, x: float, y: float, width: float, height: float, color: str, alpha: float = 1.0
) -> FancyBboxPatch:
    """Add one rounded bar centred on ``y`` to ``ax`` and return it."""
    from matplotlib.patches import FancyBboxPatch

    bar = FancyBboxPatch(
        (x, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor=color,
        alpha=alpha,
        edgecolor="none",
        zorder=3,
    )
    ax.add_patch(bar)
    return bar


def axis_bounds(values: np.ndarray, stds: np.ndarray) -> tuple[float, float]:
    """Compute padded x-axis bounds that stay finite for any input.

    Metrics can be NaN (a fold whose metric could not be computed) or negative
    (a normalized correlation below the reference model), so the bounds are
    derived from the finite values only and fall back to a unit axis when there
    are none. Without this an all-NaN column made ``set_xlim`` raise
    ``ValueError: Axis limits cannot be NaN or Inf`` and took the whole report
    down with it.

    :param values: Metric values, possibly containing NaN.
    :param stds: Matching standard deviations, possibly containing NaN.
    :returns: ``(left, right)`` limits, always finite with ``left < right``.
    """
    stds = np.nan_to_num(stds, nan=0.0)
    with np.errstate(invalid="ignore"):
        finite_high = values + stds
        finite_low = values - stds
    if not np.isfinite(finite_high).any():
        return -0.06, 1.0
    high = float(np.nanmax(finite_high))
    low = min(0.0, float(np.nanmin(finite_low)))
    span = high - low
    if not np.isfinite(span) or span <= 0:
        span = max(abs(high), 1.0)
    return low - span * 0.06, high + span * 0.18


def draw_ranked_metric_axis(
    ax: Axes,
    df: pd.DataFrame,
    *,
    metric_col: str,
    std_col: str,
    bar_height: float,
    font_adder: int,
    colors: dict[str, str],
    ascending: bool,
    xlabel: str,
    title: str,
    title_color: str,
) -> None:
    ax.set_facecolor(colors["background"])
    df_metric = df.dropna(subset=[metric_col]).sort_values(metric_col, ascending=ascending).reset_index(drop=True)
    y_positions = np.arange(len(df_metric) - 1, -1, -1)
    left, right = axis_bounds(df_metric[metric_col].to_numpy(dtype=float), df_metric[std_col].to_numpy(dtype=float))
    span = right - left

    for i, (_, row) in enumerate(df_metric.iterrows()):
        style = get_bar_color(i, row["is_baseline"])
        draw_bar(ax, 0, y_positions[i], row[metric_col], bar_height, style["color"], style["alpha"])
        label_color = style["color"] if not row["is_baseline"] else colors["text_secondary"]
        ax.text(
            row[metric_col] + span * 0.02,
            y_positions[i],
            f"{row[metric_col]:.3f}",
            va="center",
            ha="left",
            fontsize=9 + font_adder,
            fontweight="bold",
            color=label_color,
            zorder=5,
        )
        if i < 3 and not row["is_baseline"]:
            medals = ["\u2460", "\u2461", "\u2462"]
            ax.text(
                left + span * 0.02,
                y_positions[i],
                medals[i],
                va="center",
                ha="center",
                fontsize=14 + font_adder,
                fontweight="bold",
                color=style["color"],
                zorder=5,
            )

    ax.set_xlim(left, right)
    ax.set_ylim(-0.8, len(df_metric) - 0.2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_metric["algorithm"].values, fontsize=10 + font_adder)

    for i, label in enumerate(ax.get_yticklabels()):
        row = df_metric.iloc[i]
        if i < 3 and not row["is_baseline"]:
            label.set_fontweight("bold")
            label.set_color(get_bar_color(i, False)["color"])
        elif row["is_baseline"]:
            label.set_style("italic")
            label.set_color(colors["text_secondary"])
        else:
            label.set_color(colors["text"])

    ax.set_xlabel(xlabel, fontsize=12 + font_adder, fontweight="bold", labelpad=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, color=colors["grid"])
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=colors["text_secondary"])
    ax.set_title(title, fontsize=14 + font_adder, fontweight="bold", color=title_color, pad=15)


def gradient_char_colors(title_text: str) -> list[str]:
    """Return one hex color per character, running from teal through blue to purple."""
    n_chars = len(title_text)
    colors_list = []
    for j in range(n_chars):
        t = j / max(n_chars - 1, 1)
        if t < 0.5:
            t2 = t * 2
            r = int(0x14 + (0x29 - 0x14) * t2)
            g = int(0xB8 + (0xAB - 0xB8) * t2)
            b = int(0xA6 + (0xCA - 0xA6) * t2)
        else:
            t2 = (t - 0.5) * 2
            r = int(0x29 + (0x9D - 0x29) * t2)
            g = int(0xAB + (0x4E - 0xAB) * t2)
            b = int(0xCA + (0xDD - 0xCA) * t2)
        colors_list.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors_list


def draw_gradient_title(fig: Figure, title_text: str, font_adder: int, y: float = 0.97) -> None:
    """Write ``title_text`` across the figure, one character at a time, in the gradient colors."""
    title_x_start = 0.5 - len(title_text) * 0.012
    char_colors = gradient_char_colors(title_text)
    for j, char in enumerate(title_text):
        fig.text(
            title_x_start + j * 0.024,
            y,
            char,
            fontsize=24 + font_adder,
            fontweight="bold",
            color=char_colors[j],
            ha="center",
        )


def draw_subtitle(
    fig: Figure, dataset: str, measure: str, test_mode_label: str, font_adder: int, colors: dict, y: float = 0.92
) -> None:
    fig.text(
        0.5,
        y,
        f"{dataset} Dataset  \u2022  {measure}  \u2022  {test_mode_label}",
        ha="center",
        fontsize=12 + font_adder,
        color=colors["text_secondary"],
    )


def draw_legend(fig: Figure, font_adder: int, colors: dict, y: float = 0.02) -> None:
    """Add the medal, competitor and baseline legend at the bottom of the figure."""
    import matplotlib.patches as mpatches

    legend_elements = [
        mpatches.Patch(facecolor="#F4D03F", label="#1 Champion", edgecolor="none"),
        mpatches.Patch(facecolor="#BDC3C7", label="#2 Runner-up", edgecolor="none"),
        mpatches.Patch(facecolor="#E67E22", label="#3 Third Place", edgecolor="none"),
        mpatches.Patch(facecolor=COMPETITOR_COLOR, alpha=0.85, label="Competitor", edgecolor="none"),
        mpatches.Patch(facecolor="#5a5a5a", alpha=1, label="Baseline", edgecolor="none"),
    ]
    legend = fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        frameon=True,
        facecolor=colors["surface"],
        edgecolor=colors["grid"],
        fontsize=10 + font_adder,
        bbox_to_anchor=(0.5, y),
    )
    legend.get_frame().set_alpha(0.9)
    for text in legend.get_texts():
        text.set_color(colors["text"])


LEAVE_OUT_ENTITIES = {"LSO": "Sample", "LDO": "Drug", "LPO": "Pair", "LTO": "Tissue", "LStO": "Study"}


def get_test_mode_name(test_mode: str, n_folds: int) -> str:
    """Spell out a split mode for the subtitle.

    :param test_mode: Split mode, for example ``"LSO"``.
    :param n_folds: Number of folds the experiment ran. Leave-study-out has one fold
        per study, so its title leaves the count out.
    :returns: For example ``"5-Fold Leave-Sample-Out Cross Validation"``; an unknown mode is returned unchanged.
    """
    entity = LEAVE_OUT_ENTITIES.get(test_mode)
    if entity is None:
        return test_mode
    folds = "" if test_mode == "LStO" else f"{n_folds}-Fold "
    return f"{folds}Leave-{entity}-Out Cross Validation"
