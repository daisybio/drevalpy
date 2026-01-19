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
from matplotlib.patches import FancyBboxPatch

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
    """
    Configure global matplotlib parameters for the current theme.

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
    Load and aggregate results from the evaluation CSV.

    :param results_path: Path to evaluation_results.csv.
    :param test_mode: Filtering mode (e.g., LCO).
    :raises FileNotFoundError: If path does not exist.
    :raises ValueError: If no data matches criteria.
    :return: Processed DataFrame.
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


def get_bar_color(rank: int, is_baseline: bool) -> dict:
    """
    Assign colors based on model rank and type.

    :param rank: Model index in sorted list.
    :param is_baseline: Boolean if model is a baseline.
    :return: Styling dictionary.
    """
    if is_baseline:
        return {"color": "#5a5a5a", "alpha": 1.0}

    medal_gold = "#F4D03F"
    medal_silver = "#BDC3C7"
    medal_bronze = "#E67E22"

    if rank == 0:
        return {"color": medal_gold, "alpha": 1.0}
    elif rank == 1:
        return {"color": medal_silver, "alpha": 1.0}
    elif rank == 2:
        return {"color": medal_bronze, "alpha": 1.0}

    return {"color": COMPETITOR_COLOR, "alpha": 0.85}


def draw_bar(ax, x: float, y: float, width: float, height: float, color: str, alpha: float = 1.0):
    """
    Draw a custom rounded rectangle bar.

    :param ax: Matplotlib axis.
    :param x: Origin X.
    :param y: Origin Y.
    :param width: Bar width.
    :param height: Bar height.
    :param color: Hex color.
    :param alpha: Transparency.
    :return: Patch artist.
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
    measure: str = "LN_IC50_curvecurator",
    figsize: tuple = (16, 12),
    show_top_n: Optional[int] = None,
    font_adder: int = 6,
) -> tuple:
    """
    Generate the dual-panel leaderboard figure.

    :param df: Input results data.
    :param output_path: File path for save.
    :param test_mode: Evaluation mode name.
    :param dataset: Dataset name.
    :param measure: Performance measure.
    :param figsize: Figure dimensions.
    :param show_top_n: Limit displayed models.
    :param font_adder: Scale for text.
    :return: Figure and axes tuple.
    """
    configure_matplotlib(font_adder=font_adder)

    if show_top_n:
        df = df.head(show_top_n)

    n_models = len(df)
    y_positions = np.arange(n_models - 1, -1, -1)
    bar_height = 0.65

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor=COLORS["background"])
    fig.subplots_adjust(wspace=0.4)

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

    ax1.set_xlabel("Normalized PCC", fontsize=12 + font_adder, fontweight="bold", labelpad=10)
    ax1.xaxis.grid(True, linestyle="--", alpha=0.3, color=COLORS["grid"])
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="x", colors=COLORS["text_secondary"])
    ax1.set_title(
        "Normalized Pearson  ↑  higher is better",
        fontsize=14 + font_adder,
        fontweight="bold",
        color="#29ABCA",
        pad=15,
    )

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
        facecolor=COLORS["surface"],
        edgecolor=COLORS["grid"],
        fontsize=10 + font_adder,
        bbox_to_anchor=(0.5, 0.02),
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
    plt.close(fig)
    print(f"Saved leaderboard to: {output_path}")

    return fig, (ax1, ax2)


def _get_test_mode_name(test_mode: str) -> str:
    """
    Map shorthand mode codes to full descriptive names.

    :param test_mode: Suffix code (LCO, etc).
    :return: Full string name.
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
