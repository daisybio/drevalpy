#!/usr/bin/env python3
"""DrEvalPy Leaderboard Visualization.

This script generates a leaderboard visualization (normalized PCC and RMSE) from
the evaluation results CSV file produced by the DrEvalPy evaluation pipeline.
Usage:
python create_leaderboard.py --results_path /path/to/results.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .leaderboard_plot import (
    draw_bar,
    draw_footer,
    draw_gradient_title,
    draw_leaderboard_legend,
    draw_leaderboard_panels,
    draw_logo,
    draw_subtitle,
    get_bar_color,
)

__all__ = [
    "COLORS",
    "COMPETITOR_COLOR",
    "configure_matplotlib",
    "create_leaderboard",
    "draw_bar",
    "get_bar_color",
    "load_results",
    "main",
]

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

COLORS = DARK_THEME

COMPETITOR_COLOR = "#6A5ACD"


def configure_matplotlib(font_adder: int = 0):
    """Configure global matplotlib parameters for the current theme.

    :param font_adder: Increment added to the base font size.
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
    """Load and aggregate results from an evaluation CSV.

    :param results_path: Path to ``evaluation_results.csv``.
    :param test_mode: Test mode filter (for example ``"LCO"``).

    :returns: DataFrame with mean and std of normalized PCC and RMSE per algorithm.

    :raises FileNotFoundError: If ``results_path`` does not exist.
    :raises ValueError: If no rows match predictions and the test mode.
    """
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    df = pd.read_csv(path, index_col=0)
    df = df[(df["rand_setting"] == "predictions") & (df["test_mode"] == test_mode)]

    if df.empty:
        raise ValueError(f"No results found for rand_setting='predictions' and test_mode='{test_mode}'")

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


def create_leaderboard(
    df: pd.DataFrame,
    output_path: str,
    test_mode: str = "LCO",
    dataset: str = "CTRPv2",
    measure: str = "LN_IC50_curvecurator",
    figsize: tuple = (16, 12),
    show_top_n: int | None = None,
    font_adder: int = 6,
) -> tuple:
    """Generate the dual-panel leaderboard figure.

    :param df: Aggregated results per algorithm.
    :param output_path: File path for the saved image.
    :param test_mode: Evaluation mode label (for example ``"LCO"``).
    :param dataset: Dataset name shown in the subtitle.
    :param measure: Response measure shown in the subtitle.
    :param figsize: Figure dimensions in inches.
    :param show_top_n: Optional limit on the number of models displayed.
    :param font_adder: Font size increment for labels and titles.

    :returns: Tuple of the matplotlib figure and its two axes.
    """
    configure_matplotlib(font_adder=font_adder)

    if show_top_n:
        df = df.head(show_top_n)

    n_models = len(df)
    y_positions = np.arange(n_models - 1, -1, -1)
    bar_height = 0.65

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor=COLORS["background"])
    fig.subplots_adjust(wspace=0.4)
    draw_leaderboard_panels(
        fig,
        (ax1, ax2),
        df,
        y_positions=y_positions,
        bar_height=bar_height,
        font_adder=font_adder,
        colors=COLORS,
    )

    draw_gradient_title(fig, "DrEval Challenge Leaderboard", font_adder)
    draw_subtitle(fig, dataset, measure, _get_test_mode_name(test_mode), font_adder, COLORS)
    draw_logo(fig)
    draw_leaderboard_legend(fig, font_adder, COLORS)
    draw_footer(fig, font_adder, COLORS)

    plt.tight_layout(rect=(0, 0.06, 1, 0.90))
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"], transparent=False)
    plt.close(fig)
    print(f"Saved leaderboard to: {output_path}")

    return fig, (ax1, ax2)


def _get_test_mode_name(test_mode: str) -> str:
    """Map shorthand mode codes to full descriptive names.

    :param test_mode: Suffix code (for example ``LCO``).

    :returns: Full descriptive name for the test mode.
    """
    names = {
        "LCO": "10-Fold Leave-Cell-Out Cross Validation",
        "LDO": "10-Fold Leave-Drug-Out Cross Validation",
        "LPO": "10-Fold Leave-Pair-Out Cross Validation",
        "LTO": "10-Fold Leave-Tissue-Out Cross Validation",
    }
    return names.get(test_mode, test_mode)


def main():
    """Execute dual-theme leaderboard generation."""
    parser = argparse.ArgumentParser(
        description="Generate DrEvalPy leaderboard visualization (Dark & Light modes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_path", "-r", type=str, required=True, help="Path to evaluation_results.csv")
    parser.add_argument("--output_dir", "-o", type=str, default="docs/_static/img", help="Directory to save images")
    parser.add_argument("--test_mode", "-t", type=str, default="LCO", choices=["LCO", "LDO", "LPO", "LTO"])
    parser.add_argument("--dataset", "-d", type=str, default="CTRPv2", help="Dataset name")
    parser.add_argument("--measure", "-m", type=str, default="LN_IC50_curvecurator", help="Response measure")
    parser.add_argument("--top_n", "-n", type=int, default=None, help="Top N models")
    parser.add_argument("--font_adder", type=int, default=6, help="Font size increment")

    args = parser.parse_args()

    df = load_results(args.results_path, test_mode=args.test_mode)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global COLORS

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
