#!/usr/bin/env python3
"""
DrEvalPy Leaderboard Visualization.

This script generates a leaderboard visualization (normalized PCC and RMSE) from
the evaluation results CSV file produced by the DrEvalPy evaluation pipeline.
Usage:
python create_leaderboard.py --results_path /path/to/results.csv
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import FancyBboxPatch, Rectangle

# --- Theme Definitions ---
DARK_THEME = {
    "background": "#0d1117",
    "surface": "#2d2d2d",
    "text": "#ece7e4",
    "text_secondary": "#a0a0a0",
    "grid": "#30363d",
}

LIGHT_THEME = {
    "background": "#ffffff",
    "surface": "#f6f8fa",
    "text": "#1f2328",
    "text_secondary": "#57606a",
    "grid": "#d0d7de",
}

# This will be updated dynamically during the dual-generation loop
COLORS = DARK_THEME

# Gradient colors for competitors
GRADIENT_COLORS = [
    "#14b8a6",  # teal
    "#29ABCA",  # blue
    "#5B8DEE",
    "#9D4EDD",  # purple
    "#7B68EE",
    "#6A5ACD",
    "#8470FF",
]


class GradientHandler(HandlerBase):
    """Custom legend handler for gradient patches."""

    def __init__(self, colors: list):
        """
        Initialize the gradient handler.

        :param colors: List of hex colors for the gradient.
        """
        self.colors = colors
        super().__init__()

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        """
        Create the visual artists for the legend handle.

        :param legend: The legend object.
        :param orig_handle: The original handle.
        :param xdescent: The x-descent.
        :param ydescent: The y-descent.
        :param width: Width of the handle.
        :param height: Height of the handle.
        :param fontsize: Font size in pixels.
        :param trans: The transform applied to the artist.
        :return: List of matplotlib Rectangle objects.
        """
        patches = []
        n = len(self.colors)
        patch_width = width / n
        for i, color in enumerate(self.colors):
            patch = Rectangle(
                (xdescent + i * patch_width, ydescent),
                patch_width,
                height,
                facecolor=color,
                edgecolor="none",
                transform=trans,
            )
            patches.append(patch)
        return patches


def configure_matplotlib(font_adder: int = 0):
    """
    Configure matplotlib for chosen mode aesthetic.

    :param font_adder: Increment to add to the base font size.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": COLORS["background"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "grid.color": COLORS["grid"],
            "font.family": "sans-serif",
            "font.size": 11 + font_adder,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_results(results_path: str, test_mode: str = "LCO") -> pd.DataFrame:
    """
    Load evaluation results from DrEval CSV file.

    :param results_path: Path to the evaluation_results.csv file from dreval-report.
    :param test_mode: Test mode to filter for (LCO, LDO, LPO, LTO).
    :raises FileNotFoundError: If the results file does not exist at results_path.
    :raises ValueError: If no rows match the filter criteria for test_mode.
    :return: DataFrame with algorithm, PCC, PCC_std, RMSE, RMSE_std, is_baseline.
    """
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    df = pd.read_csv(path, index_col=0)

    # Filter for predictions and specified test mode
    df = df[(df["rand_setting"] == "predictions") & (df["test_mode"] == test_mode)]

    if df.empty:
        raise ValueError(f"No results found for rand_setting='predictions' and test_mode='{test_mode}'")

    # Aggregate across CV splits - take mean and std of metrics per algorithm
    df_agg = (
        df.groupby("algorithm")
        .agg(
            {
                "Pearson: normalized": ["mean", "std"],
                "RMSE": ["mean", "std"],
            }
        )
        .reset_index()
    )

    df_agg.columns = ["algorithm", "PCC", "PCC_std", "RMSE", "RMSE_std"]

    df_agg["PCC_std"] = df_agg["PCC_std"].fillna(0)
    df_agg["RMSE_std"] = df_agg["RMSE_std"].fillna(0)

    df_agg["is_baseline"] = df_agg["algorithm"].str.startswith("Naive")

    return df_agg.sort_values("PCC", ascending=False).reset_index(drop=True)


def get_bar_color(rank: int, is_baseline: bool) -> dict:
    """
    Get bar styling based on rank.

    :param rank: The zero-indexed rank of the model.
    :param is_baseline: Whether the model is a baseline model.
    :return: Dictionary containing 'color' and 'alpha' keys.
    """
    if is_baseline:
        return {"color": "#5a5a5a", "alpha": 0.5}

    medal_gold = "#F4D03F"
    medal_silver = "#BDC3C7"
    medal_bronze = "#E67E22"

    if rank == 0:
        return {"color": medal_gold, "alpha": 1.0}
    elif rank == 1:
        return {"color": medal_silver, "alpha": 1.0}
    elif rank == 2:
        return {"color": medal_bronze, "alpha": 1.0}

    idx = min(rank - 3, len(GRADIENT_COLORS) - 1)
    return {"color": GRADIENT_COLORS[idx], "alpha": 0.85}


def draw_bar(ax, x: float, y: float, width: float, height: float, color: str, alpha: float = 1.0):
    """
    Draw a rounded bar.

    :param ax: The matplotlib axes to draw on.
    :param x: X-coordinate of the bar start.
    :param y: Y-coordinate of the bar center.
    :param width: Width of the bar.
    :param height: Height of the bar.
    :param color: Hex color code.
    :param alpha: Transparency value (0-1).
    :return: The created bar patch.
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


def create_leaderboard(
    df: pd.DataFrame,
    output_path: str,
    test_mode: str = "LCO",
    dataset: str = "CTRPv2",
    measure: str = "LN_IC50",
    figsize: tuple[int, int] = (16, 12),
    show_top_n: Optional[int] = None,
    font_adder: int = 6,
) -> tuple:
    """
    Create a dual leaderboard visualization (PCC and RMSE).

    :param df: DataFrame with algorithm, PCC, RMSE, is_baseline columns.
    :param output_path: Path to save the output image.
    :param test_mode: Test mode (LCO, LDO, LPO, LTO).
    :param dataset: Dataset name.
    :param measure: Response measure.
    :param figsize: Figure size (width, height).
    :param show_top_n: Only show top N models.
    :param font_adder: Increment to add to the base font size.
    :return: A tuple of (fig, (ax1, ax2)) matplotlib objects.
    """
    configure_matplotlib(font_adder=font_adder)

    if show_top_n:
        df = df.head(show_top_n)

    n_models = len(df)
    y_positions = np.arange(n_models - 1, -1, -1)
    bar_height = 0.65

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor=COLORS["background"])
    fig.subplots_adjust(wspace=0.4)

    # LEFT PLOT: PCC
    ax1.set_facecolor(COLORS["background"])
    df_pcc = df.sort_values("PCC", ascending=False).reset_index(drop=True)
    max_pcc = (df_pcc["PCC"] + df_pcc["PCC_std"]).max() * 1.18

    for i, (_, row) in enumerate(df_pcc.iterrows()):
        style = get_bar_color(i, row["is_baseline"])
        draw_bar(ax1, 0, y_positions[i], row["PCC"], bar_height, style["color"], style["alpha"])

        label_color = style["color"] if not row["is_baseline"] else COLORS["text_secondary"]
        label_x = row["PCC"] + max_pcc * 0.02
        ax1.text(
            label_x,
            y_positions[i],
            f"{row['PCC']:.3f}",
            va="center",
            ha="left",
            fontsize=9 + font_adder,
            fontweight="bold",
            color=label_color,
            zorder=5,
        )

        if i < 3 and not row["is_baseline"]:
            medals = ["①", "②", "③"]
            ax1.text(
                -max_pcc * 0.03,
                y_positions[i],
                medals[i],
                va="center",
                ha="center",
                fontsize=14 + font_adder,
                fontweight="bold",
                color=style["color"],
                zorder=5,
            )

    ax1.set_xlim(-max_pcc * 0.06, max_pcc)
    ax1.set_ylim(-0.8, n_models - 0.2)
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(df_pcc["algorithm"].values, fontsize=10 + font_adder)

    for i, label in enumerate(ax1.get_yticklabels()):
        if i < 3 and not df_pcc.iloc[i]["is_baseline"]:
            label.set_fontweight("bold")
            label.set_color(get_bar_color(i, False)["color"])
        elif df_pcc.iloc[i]["is_baseline"]:
            label.set_style("italic")
            label.set_color(COLORS["text_secondary"])
        else:
            label.set_color(COLORS["text"])

    ax1.set_xlabel("Pearson Correlation Coefficient", fontsize=12 + font_adder, fontweight="bold", labelpad=10)
    ax1.xaxis.grid(True, linestyle="--", alpha=0.3, color=COLORS["grid"])
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="x", colors=COLORS["text_secondary"])
    ax1.set_title(
        "normalized Pearson  ↑  higher is better",
        fontsize=14 + font_adder,
        fontweight="bold",
        color="#29ABCA",
        pad=15,
    )

    # RIGHT PLOT: RMSE
    ax2.set_facecolor(COLORS["background"])
    df_rmse = df.sort_values("RMSE", ascending=True).reset_index(drop=True)
    max_rmse = (df_rmse["RMSE"] + df_rmse["RMSE_std"]).max() * 1.18

    for i, (_, row) in enumerate(df_rmse.iterrows()):
        style = get_bar_color(i, row["is_baseline"])
        draw_bar(ax2, 0, y_positions[i], row["RMSE"], bar_height, style["color"], style["alpha"])

        label_color = style["color"] if not row["is_baseline"] else COLORS["text_secondary"]
        label_x = row["RMSE"] + max_rmse * 0.02
        ax2.text(
            label_x,
            y_positions[i],
            f"{row['RMSE']:.3f}",
            va="center",
            ha="left",
            fontsize=9 + font_adder,
            fontweight="bold",
            color=label_color,
            zorder=5,
        )

        if i < 3 and not row["is_baseline"]:
            medals = ["①", "②", "③"]
            ax2.text(
                -max_rmse * 0.03,
                y_positions[i],
                medals[i],
                va="center",
                ha="center",
                fontsize=14 + font_adder,
                fontweight="bold",
                color=style["color"],
                zorder=5,
            )

    ax2.set_xlim(-max_rmse * 0.06, max_rmse)
    ax2.set_ylim(-0.8, n_models - 0.2)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(df_rmse["algorithm"].values, fontsize=10 + font_adder)
    ax2.set_xlabel("Root Mean Square Error", fontsize=12 + font_adder, fontweight="bold", labelpad=10)

    for i, label in enumerate(ax2.get_yticklabels()):
        if i < 3 and not df_rmse.iloc[i]["is_baseline"]:
            label.set_fontweight("bold")
            label.set_color(get_bar_color(i, False)["color"])
        elif df_rmse.iloc[i]["is_baseline"]:
            label.set_style("italic")
            label.set_color(COLORS["text_secondary"])
        else:
            label.set_color(COLORS["text"])

    ax2.xaxis.grid(True, linestyle="--", alpha=0.3, color=COLORS["grid"])
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="x", colors=COLORS["text_secondary"])
    ax2.set_title("RMSE  ↓  lower is better", fontsize=14 + font_adder, fontweight="bold", color="#FF6B9D", pad=15)

    # Rainbow title
    title_text = "DrEval Challenge Leaderboard"
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

    title_x_start = 0.5 - len(title_text) * 0.012
    for j, char in enumerate(title_text):
        fig.text(
            title_x_start + j * 0.024,
            0.97,
            char,
            fontsize=24 + font_adder,
            fontweight="bold",
            color=gradient_colors[j],
            ha="center",
        )
    fig.text(
        0.5,
        0.92,
        f"{dataset} Dataset  •  {measure}  •  {_get_test_mode_name(test_mode)}",
        ha="center",
        fontsize=12 + font_adder,
        color=COLORS["text_secondary"],
    )

    logo_path = Path("docs/_static/img/DrugResponseEvalLogo.svg")
    if logo_path.exists():
        try:
            from io import BytesIO

            import cairosvg
            from PIL import Image

            png_data = cairosvg.svg2png(url=str(logo_path))
            logo_img = Image.open(BytesIO(png_data))
            logo_ax = fig.add_axes((0.8, 0.94, 0.15, 0.06))
            logo_ax.imshow(logo_img)
            logo_ax.axis("off")
        except Exception as e:
            print(e)
            pass

    # Legend
    gradient_patch = mpatches.Patch(facecolor="#14b8a6", label="Competitor", edgecolor="none")
    legend_elements = [
        mpatches.Patch(facecolor="#F4D03F", label="#1 Champion", edgecolor="none"),
        mpatches.Patch(facecolor="#BDC3C7", label="#2 Runner-up", edgecolor="none"),
        mpatches.Patch(facecolor="#E67E22", label="#3 Third Place", edgecolor="none"),
        gradient_patch,
        mpatches.Patch(facecolor="#5a5a5a", alpha=0.5, label="Baseline", edgecolor="none"),
    ]
    handler_map = {gradient_patch: GradientHandler(GRADIENT_COLORS)}

    legend = fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        frameon=True,
        facecolor=COLORS["surface"],
        edgecolor=COLORS["grid"],
        fontsize=10 + font_adder,
        bbox_to_anchor=(0.5, 0.02),
        handler_map=handler_map,
    )
    legend.get_frame().set_alpha(0.9)
    for text in legend.get_texts():
        text.set_color(COLORS["text"])

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
        color=COLORS["text_secondary"],
        style="italic",
        linespacing=1.0,
    )

    plt.tight_layout(rect=(0, 0.06, 1, 0.90))
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"], transparent=False)
    plt.close(fig)  # Close to prevent memory accumulation in loop
    print(f"Saved leaderboard to: {output_path}")

    return fig, (ax1, ax2)


def _get_test_mode_name(test_mode: str) -> str:
    """
    Get full name for test mode.

    :param test_mode: The short code for the test mode.
    :return: The human-readable description.
    """
    names = {
        "LCO": "10-Fold Leave-Cell-Out Cross Validation",
        "LDO": "10-Fold Leave-Drug-Out Cross Validation",
        "LPO": "10-Fold Leave-Pair-Out Cross Validation",
        "LTO": "10-Fold Leave-Tissue-Out Cross Validation",
    }
    return names.get(test_mode, test_mode)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate DrEvalPy leaderboard visualization (Dark & Light modes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_path", "-r", type=str, required=True, help="Path to evaluation_results.csv")
    parser.add_argument("--output_dir", "-o", type=str, default="docs/_static/img", help="Directory to save images")
    parser.add_argument("--test_mode", "-t", type=str, default="LCO", choices=["LCO", "LDO", "LPO", "LTO"])
    parser.add_argument("--dataset", "-d", type=str, default="CTRPv2", help="Dataset name")
    parser.add_argument("--measure", "-m", type=str, default="LN_IC50", help="Response measure")
    parser.add_argument("--top_n", "-n", type=int, default=None, help="Top N models")
    parser.add_argument("--font_adder", type=int, default=6, help="Font size increment")

    args = parser.parse_args()

    # Load data once
    df = load_results(args.results_path, test_mode=args.test_mode)

    # Setup output paths
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate both
    global COLORS

    # DARK MODE
    COLORS = DARK_THEME
    create_leaderboard(
        df=df,
        output_path=str(out_dir / "leaderboard_dark.png"),
        test_mode=args.test_mode,
        dataset=args.dataset,
        measure=args.measure,
        show_top_n=args.top_n,
        font_adder=args.font_adder,
    )

    # LIGHT MODE
    COLORS = LIGHT_THEME
    create_leaderboard(
        df=df,
        output_path=str(out_dir / "leaderboard_light.png"),
        test_mode=args.test_mode,
        dataset=args.dataset,
        measure=args.measure,
        show_top_n=args.top_n,
        font_adder=args.font_adder,
    )


if __name__ == "__main__":
    main()
