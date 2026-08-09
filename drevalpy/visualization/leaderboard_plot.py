"""Drawing primitives for DrEval leaderboard figures."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from upath import UPath as Path

COMPETITOR_COLOR = "#6A5ACD"


def get_bar_color(rank: int, is_baseline: bool) -> dict[str, Any]:
    """Assign bar colors based on model rank and baseline status.

    :param rank: Zero-based rank in the sorted leaderboard.
    :param is_baseline: Whether the model is a naive baseline.

    :returns: Dict with ``color`` and ``alpha`` keys for matplotlib styling.
    """
    if is_baseline:
        return {"color": "#5a5a5a", "alpha": 1.0}
    medal_colors = ["#F4D03F", "#BDC3C7", "#E67E22"]
    if rank < len(medal_colors):
        return {"color": medal_colors[rank], "alpha": 1.0}
    return {"color": COMPETITOR_COLOR, "alpha": 0.85}


def draw_bar(
    ax,
    x: float,
    y: float | int | np.integer,
    width: float,
    height: float,
    color: str,
    alpha: float = 1.0,
):
    """Draw a custom rounded rectangle bar.

    :param ax: Matplotlib axes to draw on.
    :param x: Left edge of the bar.
    :param y: Center y-coordinate of the bar.
    :param width: Bar width.
    :param height: Bar height.
    :param color: Face color.
    :param alpha: Bar transparency.

    :returns: The added ``FancyBboxPatch`` instance.
    """
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


def _style_yticklabels(ax, df_sorted: pd.DataFrame, colors: dict[str, str], font_adder: int) -> None:
    for i, label in enumerate(ax.get_yticklabels()):
        row = df_sorted.iloc[i]
        if i < 3 and not row["is_baseline"]:
            label.set_fontweight("bold")
            label.set_color(get_bar_color(i, False)["color"])
        elif row["is_baseline"]:
            label.set_style("italic")
            label.set_color(colors["text_secondary"])
        else:
            label.set_color(colors["text"])


def _draw_ranked_metric_axis(
    ax,
    df: pd.DataFrame,
    *,
    metric_col: str,
    std_col: str,
    y_positions: np.ndarray,
    bar_height: float,
    font_adder: int,
    colors: dict[str, str],
    ascending: bool,
    xlabel: str,
    title: str,
    title_color: str,
) -> None:
    ax.set_facecolor(colors["background"])
    df_metric = df.sort_values(metric_col, ascending=ascending).reset_index(drop=True)
    max_val = (df_metric[metric_col] + df_metric[std_col]).max() * 1.18

    for i, (_, row) in enumerate(df_metric.iterrows()):
        style = get_bar_color(i, row["is_baseline"])
        draw_bar(ax, 0, y_positions[i], row[metric_col], bar_height, style["color"], style["alpha"])
        label_color = style["color"] if not row["is_baseline"] else colors["text_secondary"]
        ax.text(
            row[metric_col] + max_val * 0.02,
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
            medals = ["①", "②", "③"]
            ax.text(
                -max_val * 0.03,
                y_positions[i],
                medals[i],
                va="center",
                ha="center",
                fontsize=14 + font_adder,
                fontweight="bold",
                color=style["color"],
                zorder=5,
            )

    ax.set_xlim(-max_val * 0.06, max_val)
    ax.set_ylim(-0.8, len(df) - 0.2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_metric["algorithm"].values, fontsize=10 + font_adder)
    _style_yticklabels(ax, df_metric, colors, font_adder)
    ax.set_xlabel(xlabel, fontsize=12 + font_adder, fontweight="bold", labelpad=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, color=colors["grid"])
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=colors["text_secondary"])
    ax.set_title(title, fontsize=14 + font_adder, fontweight="bold", color=title_color, pad=15)


def _gradient_char_colors(title_text: str) -> list[str]:
    n_chars = len(title_text)
    gradient_colors = []
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
        gradient_colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return gradient_colors


def draw_gradient_title(fig, title_text: str, font_adder: int) -> None:
    """Draw a multi-color gradient title on the figure.

    :param fig: Matplotlib figure.
    :param title_text: Title string to render.
    :param font_adder: Font size increment.
    """
    title_x_start = 0.5 - len(title_text) * 0.012
    for j, char in enumerate(title_text):
        fig.text(
            title_x_start + j * 0.024,
            0.97,
            char,
            fontsize=24 + font_adder,
            fontweight="bold",
            color=_gradient_char_colors(title_text)[j],
            ha="center",
        )


def draw_subtitle(
    fig, dataset: str, measure: str, test_mode_label: str, font_adder: int, colors: dict[str, str]
) -> None:
    """Draw dataset, measure, and test-mode subtitle text.

    :param fig: Matplotlib figure.
    :param dataset: Dataset name for the subtitle.
    :param measure: Response measure for the subtitle.
    :param test_mode_label: Human-readable test mode label.
    :param font_adder: Font size increment.
    :param colors: Theme color mapping.
    """
    fig.text(
        0.5,
        0.92,
        f"{dataset} Dataset  •  {measure}  •  {test_mode_label}",
        ha="center",
        fontsize=12 + font_adder,
        color=colors["text_secondary"],
    )


def draw_logo(fig) -> None:
    """Embed the DrugResponseEval logo when the SVG asset is available.

    :param fig: Matplotlib figure.
    """
    logo_path = Path("docs/_static/img/DrugResponseEvalLogo.svg")
    if not logo_path.exists():
        return
    try:
        import cairosvg
        from PIL import Image

        png_data = cairosvg.svg2png(url=str(logo_path))
        logo_img = Image.open(BytesIO(png_data))
        logo_ax = fig.add_axes((0.8, 0.94, 0.15, 0.06))
        logo_ax.imshow(logo_img)
        logo_ax.axis("off")
    except Exception as exc:
        print(exc)


def draw_leaderboard_legend(fig, font_adder: int, colors: dict[str, str]) -> None:
    """Draw the rank and baseline legend below the leaderboard.

    :param fig: Matplotlib figure.
    :param font_adder: Font size increment.
    :param colors: Theme color mapping.
    """
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
        bbox_to_anchor=(0.5, 0.02),
    )
    legend.get_frame().set_alpha(0.9)
    for text in legend.get_texts():
        text.set_color(colors["text"])


def draw_footer(fig, font_adder: int, colors: dict[str, str]) -> None:
    """Draw submission instructions in the figure footer.

    :param fig: Matplotlib figure.
    :param font_adder: Font size increment.
    :param colors: Theme color mapping.
    """
    footer_text = (
        "Submit your model → https://drevalpy.readthedocs.io/en/latest/. "
        "Send us your results.\n\n"
        "If you significantly outperform the RandomForest, we send you chocolate!"
    )
    fig.text(
        0.5,
        -0.01,
        footer_text,
        ha="center",
        va="top",
        fontsize=14 + font_adder,
        color=colors["text_secondary"],
        style="italic",
        linespacing=1.0,
    )


def draw_leaderboard_panels(
    fig,
    axes,
    df: pd.DataFrame,
    *,
    y_positions: np.ndarray,
    bar_height: float,
    font_adder: int,
    colors: dict[str, str],
) -> None:
    """Draw normalized PCC and RMSE panels on a leaderboard figure.

    :param fig: Matplotlib figure.
    :param axes: Tuple of two metric axes.
    :param df: Aggregated leaderboard results per algorithm.
    :param y_positions: Vertical bar positions.
    :param bar_height: Height of each bar.
    :param font_adder: Font size increment.
    :param colors: Theme color mapping.
    """
    ax1, ax2 = axes
    _draw_ranked_metric_axis(
        ax1,
        df,
        metric_col="PCC",
        std_col="PCC_std",
        y_positions=y_positions,
        bar_height=bar_height,
        font_adder=font_adder,
        colors=colors,
        ascending=False,
        xlabel="Normalized PCC",
        title="Normalized Pearson  ↑  higher is better",
        title_color="#29ABCA",
    )
    _draw_ranked_metric_axis(
        ax2,
        df,
        metric_col="RMSE",
        std_col="RMSE_std",
        y_positions=y_positions,
        bar_height=bar_height,
        font_adder=font_adder,
        colors=colors,
        ascending=True,
        xlabel="Root Mean Square Error",
        title="RMSE  ↓  lower is better",
        title_color="#FF6B9D",
    )
