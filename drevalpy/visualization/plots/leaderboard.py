"""Leaderboard visualization (Matplotlib via ImageVisualization)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from drevalpy.visualization.base import ImageVisualization
from drevalpy.visualization.registry import visualization_registry
from drevalpy.visualization.requirements import PlotRequirement

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult

# --- Theme ---
DARK_THEME = {
    "background": "#0d1117",
    "surface": "#2d2d2d",
    "text": "#ece7e4",
    "text_secondary": "#a0a0a0",
    "grid": "#30363d",
}

COMPETITOR_COLOR = "#6A5ACD"


def _get_bar_color(rank: int, is_baseline: bool) -> dict[str, Any]:
    if is_baseline:
        return {"color": "#5a5a5a", "alpha": 1.0}
    medal_colors = ["#F4D03F", "#BDC3C7", "#E67E22"]
    if rank < len(medal_colors):
        return {"color": medal_colors[rank], "alpha": 1.0}
    return {"color": COMPETITOR_COLOR, "alpha": 0.85}


def _draw_bar(ax, x: float, y: float, width: float, height: float, color: str, alpha: float = 1.0):
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
        style = _get_bar_color(i, row["is_baseline"])
        _draw_bar(ax, 0, y_positions[i], row[metric_col], bar_height, style["color"], style["alpha"])
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
            medals = ["\u2460", "\u2461", "\u2462"]
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

    for i, label in enumerate(ax.get_yticklabels()):
        row = df_metric.iloc[i]
        if i < 3 and not row["is_baseline"]:
            label.set_fontweight("bold")
            label.set_color(_get_bar_color(i, False)["color"])
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


def _gradient_char_colors(title_text: str) -> list[str]:
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


def _draw_gradient_title(fig, title_text: str, font_adder: int) -> None:
    title_x_start = 0.5 - len(title_text) * 0.012
    char_colors = _gradient_char_colors(title_text)
    for j, char in enumerate(title_text):
        fig.text(
            title_x_start + j * 0.024,
            0.97,
            char,
            fontsize=24 + font_adder,
            fontweight="bold",
            color=char_colors[j],
            ha="center",
        )


def _draw_subtitle(fig, dataset: str, measure: str, test_mode_label: str, font_adder: int, colors: dict) -> None:
    fig.text(
        0.5,
        0.92,
        f"{dataset} Dataset  \u2022  {measure}  \u2022  {test_mode_label}",
        ha="center",
        fontsize=12 + font_adder,
        color=colors["text_secondary"],
    )


def _draw_legend(fig, font_adder: int, colors: dict) -> None:
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


def _get_test_mode_name(test_mode: str) -> str:
    names = {
        "LCO": "10-Fold Leave-Cell-Out Cross Validation",
        "LDO": "10-Fold Leave-Drug-Out Cross Validation",
        "LPO": "10-Fold Leave-Pair-Out Cross Validation",
        "LTO": "10-Fold Leave-Tissue-Out Cross Validation",
    }
    return names.get(test_mode, test_mode)


def _build_leaderboard_df(result: ExperimentResult) -> pd.DataFrame:
    """Build aggregated leaderboard DataFrame from an ExperimentResult."""
    rows: list[dict] = []
    for model in result.models:
        for run in model.runs:
            if run.randomization is not None:
                continue
            rows.append(
                {
                    "algorithm": run.model_name,
                    "Pearson: normalized": run.metrics.get("Pearson: normalized", float("nan")),
                    "RMSE": run.metrics.get("RMSE", float("nan")),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["algorithm", "PCC", "PCC_std", "RMSE", "RMSE_std", "is_baseline"])

    df_agg = (
        df.groupby("algorithm").agg({"Pearson: normalized": ["mean", "std"], "RMSE": ["mean", "std"]}).reset_index()
    )
    df_agg.columns = ["algorithm", "PCC", "PCC_std", "RMSE", "RMSE_std"]
    df_agg["PCC_std"] = df_agg["PCC_std"].fillna(0)
    df_agg["RMSE_std"] = df_agg["RMSE_std"].fillna(0)
    df_agg["is_baseline"] = df_agg["algorithm"].str.startswith("Naive")
    return df_agg.sort_values("PCC", ascending=False).reset_index(drop=True)


@visualization_registry.register(
    "leaderboard",
    "Leaderboard visualization of normalized PCC and RMSE rankings",
    requirements=frozenset({PlotRequirement.MULTIPLE_MODELS, PlotRequirement.MULTIPLE_FOLDS}),
)
class LeaderboardVisualization(ImageVisualization):
    """Dual-panel leaderboard using Matplotlib."""

    def __init__(self) -> None:
        """Initialize with empty state."""
        self._result: ExperimentResult | None = None
        self._fig: plt.Figure | None = None

    def compute(self, result: ExperimentResult, dataset=None) -> None:
        """Compute leaderboard rankings and create figure.

        :param result: Experiment result with multiple models and folds.
        """
        self._result = result
        self._fig = self._create_figure()

    def _create_figure(self) -> plt.Figure:
        """Create the leaderboard panels figure."""
        result = self._result
        colors = DARK_THEME
        font_adder = 6

        df = _build_leaderboard_df(result)
        if df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "No data available for leaderboard", ha="center", va="center")
            return fig

        plt.rcParams.update(
            {
                "figure.facecolor": colors["background"],
                "axes.facecolor": colors["background"],
                "axes.edgecolor": colors["grid"],
                "axes.labelcolor": colors["text"],
                "text.color": colors["text"],
                "xtick.color": colors["text"],
                "ytick.color": colors["text"],
                "grid.color": colors["grid"],
                "font.family": "sans-serif",
                "font.size": 11 + font_adder,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )

        n_models = len(df)
        y_positions = np.arange(n_models - 1, -1, -1)
        bar_height = 0.65

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12), facecolor=colors["background"])
        fig.subplots_adjust(wspace=0.4)

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
            title="Normalized Pearson  \u2191  higher is better",
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
            title="RMSE  \u2193  lower is better",
            title_color="#FF6B9D",
        )

        _draw_gradient_title(fig, "DrEval Challenge Leaderboard", font_adder)
        _draw_subtitle(
            fig,
            result.dataset_name if hasattr(result, "dataset_name") else "Dataset",
            "LN_IC50",
            _get_test_mode_name(result.split_mode if hasattr(result, "split_mode") else "LCO"),
            font_adder,
            colors,
        )
        _draw_legend(fig, font_adder, colors)

        plt.tight_layout(rect=(0, 0.06, 1, 0.90))

        return fig
