"""Leaderboard visualization (Matplotlib via ImageVisualization).

``matplotlib`` is imported inside the drawing helpers rather than at module
scope: ``drevalpy.registry`` imports every builtin visualization on
``import drevalpy``, so a module-scope import would put the whole pyplot stack on
the startup path of every CLI invocation. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from drevalpy.log import get_logger
from drevalpy.registry.visualization import register
from drevalpy.visualization._metric_names import holds_normalized_values, metric_keys, resolve_metric_key
from drevalpy.visualization.base import ImageVisualization
from drevalpy.visualization.requirements import PlotRequirement

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

    from drevalpy.types.results import ExperimentResult

logger = get_logger(__name__)

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


def _axis_bounds(values: np.ndarray, stds: np.ndarray) -> tuple[float, float]:
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


def _draw_ranked_metric_axis(
    ax,
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
    left, right = _axis_bounds(df_metric[metric_col].to_numpy(dtype=float), df_metric[std_col].to_numpy(dtype=float))
    span = right - left

    for i, (_, row) in enumerate(df_metric.iterrows()):
        style = _get_bar_color(i, row["is_baseline"])
        _draw_bar(ax, 0, y_positions[i], row[metric_col], bar_height, style["color"], style["alpha"])
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


def _draw_gradient_title(fig, title_text: str, font_adder: int, y: float = 0.97) -> None:
    title_x_start = 0.5 - len(title_text) * 0.012
    char_colors = _gradient_char_colors(title_text)
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


def _draw_subtitle(
    fig, dataset: str, measure: str, test_mode_label: str, font_adder: int, colors: dict, y: float = 0.92
) -> None:
    fig.text(
        0.5,
        y,
        f"{dataset} Dataset  \u2022  {measure}  \u2022  {test_mode_label}",
        ha="center",
        fontsize=12 + font_adder,
        color=colors["text_secondary"],
    )


def _draw_legend(fig, font_adder: int, colors: dict, y: float = 0.02) -> None:
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


def _get_test_mode_name(test_mode: str) -> str:
    names = {
        "LCO": "10-Fold Leave-Cell-Out Cross Validation",
        "LDO": "10-Fold Leave-Drug-Out Cross Validation",
        "LPO": "10-Fold Leave-Pair-Out Cross Validation",
        "LTO": "10-Fold Leave-Tissue-Out Cross Validation",
    }
    return names.get(test_mode, test_mode)


def _build_leaderboard_df(result: ExperimentResult) -> pd.DataFrame:
    """Build aggregated leaderboard DataFrame from an ExperimentResult.

    ``"Pearson"`` is resolved through :func:`resolve_metric_key`, so the panel
    shows the normalized correlation on a normalized experiment and the raw one
    otherwise, instead of an all-NaN column when the suffixed key is absent.

    :param result: Experiment to rank.
    :returns: One row per model with ``PCC``/``RMSE`` means and standard deviations.
    """
    import pandas as pd

    available = metric_keys(result)
    pcc_key = resolve_metric_key(available, "Pearson")
    rmse_key = resolve_metric_key(available, "RMSE")
    rows: list[dict] = []
    for model in result.models:
        for run in model.runs:
            if run.randomization is not None:
                continue
            rows.append(
                {
                    "algorithm": run.model_name,
                    "PCC": run.metrics.get(pcc_key, float("nan")) if pcc_key else float("nan"),
                    "RMSE": run.metrics.get(rmse_key, float("nan")) if rmse_key else float("nan"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["algorithm", "PCC", "PCC_std", "RMSE", "RMSE_std", "is_baseline"])

    df_agg = df.groupby("algorithm").agg({"PCC": ["mean", "std"], "RMSE": ["mean", "std"]}).reset_index()
    df_agg.columns = ["algorithm", "PCC", "PCC_std", "RMSE", "RMSE_std"]
    df_agg["PCC_std"] = df_agg["PCC_std"].fillna(0)
    df_agg["RMSE_std"] = df_agg["RMSE_std"].fillna(0)
    df_agg["is_baseline"] = df_agg["algorithm"].str.startswith("Naive")
    return df_agg.sort_values("PCC", ascending=False).reset_index(drop=True)


def _figure_geometry(n_models: int) -> tuple[float, int, float]:
    """Size the figure so every model keeps a legible tick label.

    The panels carry one tick label per model, so a fixed 12-inch canvas turns
    into an unreadable smear past roughly 20 models - the 96-model production
    report is the case that matters. Height grows linearly with the model count
    and the font shrinks back towards its base size as the list gets long.

    :param n_models: Number of models being ranked.
    :returns: ``(height_inches, font_size_offset, bar_height)``.
    """
    height = min(max(12.0, 1.4 + 0.34 * n_models), 60.0)
    font_adder = 6 if n_models <= 20 else (3 if n_models <= 50 else 1)
    return height, font_adder, 0.65


def _pcc_is_normalized(result: ExperimentResult) -> bool:
    """Whether the leaderboard's PCC column holds reference-normalized values.

    :param result: Experiment the column was built from.
    :returns: True when the values are normalized against a reference model.
    """
    key = resolve_metric_key(metric_keys(result), "Pearson")
    return key is not None and holds_normalized_values(result, key)


@register(
    "leaderboard",
    "Leaderboard visualization of normalized PCC and RMSE rankings",
    requirements=frozenset({PlotRequirement.MULTIPLE_MODELS, PlotRequirement.MULTIPLE_FOLDS}),
)
class LeaderboardVisualization(ImageVisualization):
    """Dual-panel leaderboard using Matplotlib."""

    def __init__(self) -> None:
        """Initialize with empty state."""
        self._result: ExperimentResult | None = None
        self._fig: Figure | None = None

    def compute(self, result: ExperimentResult, dataset=None) -> None:
        """Compute leaderboard rankings and create figure.

        :param result: Experiment result with multiple models and folds.
        """
        self._result = result
        self._fig = self._create_figure()

    def _create_figure(self) -> Figure:
        """Create the leaderboard panels figure."""
        import matplotlib
        from matplotlib.figure import Figure

        result = self._result
        colors = DARK_THEME

        df = _build_leaderboard_df(result)
        if df.empty or not np.isfinite(df[["PCC", "RMSE"]].to_numpy(dtype=float)).any():
            logger.warning("leaderboard: no finite Pearson/RMSE values to rank; emitting a placeholder panel")
            fig = Figure(figsize=(10, 4))
            ax = fig.add_subplot()
            ax.text(0.5, 0.5, "No data available for leaderboard", ha="center", va="center")
            return fig

        normalized = _pcc_is_normalized(result)
        n_ranked = int(df[["PCC", "RMSE"]].notna().any(axis=1).sum())
        # One tick label per model, so a 96-model experiment needs ~4x the height a
        # 12-model one does or the names overlap into an unreadable smear.
        fig_height, font_adder, bar_height = _figure_geometry(n_ranked)

        matplotlib.rcParams.update(
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

        fig = Figure(figsize=(16, fig_height), facecolor=colors["background"])
        ax1, ax2 = fig.subplots(1, 2)
        fig.subplots_adjust(wspace=0.4)

        pcc_label = "Normalized PCC" if normalized else "PCC"
        _draw_ranked_metric_axis(
            ax1,
            df,
            metric_col="PCC",
            std_col="PCC_std",
            bar_height=bar_height,
            font_adder=font_adder,
            colors=colors,
            ascending=False,
            xlabel=pcc_label,
            title=f"{'Normalized ' if normalized else ''}Pearson  \u2191  higher is better",
            title_color="#29ABCA",
        )
        _draw_ranked_metric_axis(
            ax2,
            df,
            metric_col="RMSE",
            std_col="RMSE_std",
            bar_height=bar_height,
            font_adder=font_adder,
            colors=colors,
            ascending=True,
            xlabel="Root Mean Square Error",
            title="RMSE  \u2193  lower is better",
            title_color="#FF6B9D",
        )

        # Header and footer are placed in figure fractions but should occupy a constant
        # number of inches, or a 96-model figure reserves a third of its canvas for them.
        header = 1.8 / fig_height
        footer = 0.9 / fig_height
        _draw_gradient_title(fig, "DrEval Challenge Leaderboard", font_adder, y=1 - header * 0.28)
        _draw_subtitle(
            fig,
            result.dataset_name if hasattr(result, "dataset_name") else "Dataset",
            # Every run trains on ``Dataset.response_matrix``, i.e. the response
            # modality's X, which curation fills with pEC50.
            "pEC50",
            _get_test_mode_name(result.split_mode if hasattr(result, "split_mode") else "LCO"),
            font_adder,
            colors,
            y=1 - header * 0.75,
        )
        _draw_legend(fig, font_adder, colors, y=footer * 0.25)

        fig.tight_layout(rect=(0, footer, 1, 1 - header))

        return fig
