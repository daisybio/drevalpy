"""Drawing helpers of the leaderboard: theme, ranked bar panels, title and legend.

``matplotlib`` is imported inside the drawing helpers rather than at module scope:
``drevalpy.registry`` imports every builtin visualization on ``import drevalpy``, so a
module-scope import would put the whole pyplot stack on the startup path of every CLI
invocation. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# --- Theme ---
DARK_THEME = {
    "background": "#0d1117",
    "surface": "#2d2d2d",
    "text": "#ece7e4",
    "text_secondary": "#a0a0a0",
    "grid": "#30363d",
}

COMPETITOR_COLOR = "#6A5ACD"
BASELINE_COLOR = "#5a5a5a"
#: Bar colors of the first three ranks: gold, silver, bronze.
MEDAL_COLORS = ("#F4D03F", "#BDC3C7", "#E67E22")

#: Height of a bar in data units, where the ranks are one unit apart.
_BAR_HEIGHT = 0.65


@dataclass(frozen=True)
class Panel:
    """One ranked bar panel: the table column it ranks and how it is labelled."""

    column: str
    ascending: bool
    xlabel: str
    title: str
    title_color: str


def get_bar_color(rank: int, is_baseline: bool) -> dict[str, Any]:
    """Return the bar color and opacity: grey for baselines, medals for the top three, purple otherwise."""
    if is_baseline:
        return {"color": BASELINE_COLOR, "alpha": 1.0}
    if rank < len(MEDAL_COLORS):
        return {"color": MEDAL_COLORS[rank], "alpha": 1.0}
    return {"color": COMPETITOR_COLOR, "alpha": 0.85}


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


def draw_ranked_metric_axis(ax: Axes, df: pd.DataFrame, panel: Panel, font_adder: int) -> None:
    """Draw the models ranked on ``panel.column`` as horizontal bars, with their values and medals.

    :param ax: Axes to draw on.
    :param df: The leaderboard table; models without a value in the panel's column are left out.
    :param panel: The column to rank and how the panel is labelled.
    :param font_adder: Points added to the base font sizes.
    """
    from matplotlib.patches import FancyBboxPatch

    colors = DARK_THEME
    metric_col = panel.column
    df_metric = df.dropna(subset=[metric_col]).sort_values(metric_col, ascending=panel.ascending).reset_index(drop=True)
    y_positions = np.arange(len(df_metric) - 1, -1, -1)
    left, right = axis_bounds(
        df_metric[metric_col].to_numpy(dtype=float), df_metric[f"{metric_col}_std"].to_numpy(dtype=float)
    )
    span = right - left

    for i, (_, row) in enumerate(df_metric.iterrows()):
        style = get_bar_color(i, row["is_baseline"])
        ax.add_patch(
            FancyBboxPatch(
                (0, y_positions[i] - _BAR_HEIGHT / 2),
                row[metric_col],
                _BAR_HEIGHT,
                boxstyle="round,pad=0.01,rounding_size=0.015",
                facecolor=style["color"],
                alpha=style["alpha"],
                edgecolor="none",
                zorder=3,
            )
        )
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

    ax.set_xlabel(panel.xlabel, fontsize=12 + font_adder, fontweight="bold", labelpad=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, color=colors["grid"])
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=colors["text_secondary"])
    ax.set_title(panel.title, fontsize=14 + font_adder, fontweight="bold", color=panel.title_color, pad=15)


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


def draw_legend(fig: Figure, font_adder: int, y: float = 0.02) -> None:
    """Add the medal, competitor and baseline legend at the bottom of the figure."""
    import matplotlib.patches as mpatches

    colors = DARK_THEME
    medal_labels = ("#1 Champion", "#2 Runner-up", "#3 Third Place")
    legend_elements = [
        *(
            mpatches.Patch(facecolor=color, label=label, edgecolor="none")
            for color, label in zip(MEDAL_COLORS, medal_labels, strict=True)
        ),
        mpatches.Patch(facecolor=COMPETITOR_COLOR, alpha=0.85, label="Competitor", edgecolor="none"),
        mpatches.Patch(facecolor=BASELINE_COLOR, alpha=1, label="Baseline", edgecolor="none"),
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
