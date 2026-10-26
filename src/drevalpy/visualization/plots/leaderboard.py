"""Leaderboard visualization (Matplotlib via ImageVisualization).

The panels are the four of the 1.6.1 leaderboard: normalized Pearson, RMSE, raw Pearson and
per-drug Pearson. Raw Pearson is left out of an experiment that was not normalized, where it
would repeat the first panel, and per-drug Pearson when no model has a defined value. The
table behind them is built in :mod:`drevalpy.visualization.plots._leaderboard_table`, the
drawing in :mod:`drevalpy.visualization.plots._leaderboard_draw`.

``matplotlib`` is imported inside the drawing helpers rather than at module scope:
``drevalpy.registry`` imports every builtin visualization on ``import drevalpy``, so a
module-scope import would put the whole pyplot stack on the startup path of every CLI
invocation. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from drevalpy.log import get_logger
from drevalpy.registry.visualization import register
from drevalpy.visualization.base import ImageVisualization
from drevalpy.visualization.plots._leaderboard_draw import (
    DARK_THEME,
    draw_gradient_title,
    draw_legend,
    draw_ranked_metric_axis,
    draw_subtitle,
    get_test_mode_name,
)
from drevalpy.visualization.plots._leaderboard_table import build_leaderboard_df
from drevalpy.visualization.requirements import PlotRequirement

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

    from drevalpy.types.results import ExperimentResult

logger = get_logger(__name__)


@dataclass(frozen=True)
class _Panel:
    """One ranked bar panel: the table column it ranks and how it is labelled."""

    column: str
    ascending: bool
    xlabel: str
    title: str
    title_color: str


def _figure_geometry(n_models: int) -> tuple[float, int, float]:
    """Size the figure so every model keeps a legible tick label.

    The panels carry one tick label per model, so a fixed 12-inch canvas turns
    into an unreadable smear past roughly 20 models - the 96-model production
    report is the case that matters. Height grows linearly with the model count
    and the font shrinks back towards its base size as the list gets long.

    :param n_models: Number of models being ranked.
    :returns: ``(height_inches_per_row_of_panels, font_size_offset, bar_height)``.
    """
    height = min(max(12.0, 1.4 + 0.34 * n_models), 60.0)
    font_adder = 6 if n_models <= 20 else (3 if n_models <= 50 else 1)
    return height, font_adder, 0.65


def _panels(df: pd.DataFrame, normalized: bool) -> list[_Panel]:
    """Choose the panels to draw, in reading order.

    :param df: The leaderboard table.
    :param normalized: Whether the experiment was normalized against a reference model.
    :returns: Normalized (or plain) Pearson and RMSE always, then raw Pearson if the experiment was normalized
        and has raw values, then per-drug Pearson if any model has one.
    """
    panels = [
        _Panel(
            "PCC",
            False,
            "Normalized PCC" if normalized else "PCC",
            f"{'Normalized ' if normalized else ''}Pearson  \u2191  higher is better",
            "#29ABCA",
        ),
        _Panel("RMSE", True, "Root Mean Square Error", "RMSE  \u2193  lower is better", "#FF6B9D"),
    ]
    if normalized and df["Pearson"].notna().any():
        panels.append(_Panel("Pearson", False, "Raw PCC", "Raw PCC  \u2191  higher is better", "#14B8A6"))
    if df["Pearson_per_drug"].notna().any():
        panels.append(
            _Panel(
                "Pearson_per_drug",
                False,
                "Per-drug Pearson",
                "Per-drug Pearson  \u2191  higher is better",
                "#F59E0B",
            )
        )
    return panels


@register(
    "leaderboard",
    "Leaderboard visualization of normalized PCC, RMSE, raw and per-drug Pearson rankings",
    requirements=frozenset({PlotRequirement.MULTIPLE_MODELS, PlotRequirement.MULTIPLE_FOLDS}),
)
class LeaderboardVisualization(ImageVisualization):
    """Leaderboard of up to four ranked panels using Matplotlib."""

    def __init__(self) -> None:
        """Initialize with empty state."""
        self._result: ExperimentResult | None = None
        self._fig: Figure | None = None

    def compute(self, result: ExperimentResult) -> None:
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

        df = build_leaderboard_df(result)
        metric_columns = df[["PCC", "RMSE"]].to_numpy(dtype=float) if not df.empty else np.empty((0, 2))
        if not np.isfinite(metric_columns).any():
            logger.warning("leaderboard: no finite Pearson/RMSE values to rank; emitting a placeholder panel")
            fig = Figure(figsize=(10, 4))
            ax = fig.add_subplot()
            ax.text(0.5, 0.5, "No data available for leaderboard", ha="center", va="center")
            return fig

        normalized = result.normalized_by is not None
        panels = _panels(df, normalized)
        n_rows = 1 if len(panels) <= 2 else 2
        n_ranked = int(df[[panel.column for panel in panels]].notna().any(axis=1).sum())
        # One tick label per model, so a 96-model experiment needs ~4x the height a
        # 12-model one does or the names overlap into an unreadable smear.
        row_height, font_adder, bar_height = _figure_geometry(n_ranked)
        fig_height = row_height * n_rows

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
        axes = np.atleast_1d(fig.subplots(n_rows, 2)).ravel()
        fig.subplots_adjust(wspace=0.4, hspace=0.3)
        for ax in axes[len(panels) :]:
            ax.axis("off")
        for ax, panel in zip(axes, panels, strict=False):
            draw_ranked_metric_axis(
                ax,
                df,
                metric_col=panel.column,
                std_col=f"{panel.column}_std",
                bar_height=bar_height,
                font_adder=font_adder,
                colors=colors,
                ascending=panel.ascending,
                xlabel=panel.xlabel,
                title=panel.title,
                title_color=panel.title_color,
            )

        # Header and footer are placed in figure fractions but should occupy a constant
        # number of inches, or a 96-model figure reserves a third of its canvas for them.
        header = 1.8 / fig_height
        footer = 0.9 / fig_height
        draw_gradient_title(fig, "DrEval Challenge Leaderboard", font_adder, y=1 - header * 0.28)
        draw_subtitle(
            fig,
            result.dataset_name,
            # Every run trains on ``Dataset.response_matrix``, i.e. the response
            # modality's X, which curation fills with pEC50.
            "pEC50",
            get_test_mode_name(result.split_mode, result.max_folds),
            font_adder,
            colors,
            y=1 - header * 0.75,
        )
        draw_legend(fig, font_adder, colors, y=footer * 0.25)

        fig.tight_layout(rect=(0, footer, 1, 1 - header))

        return fig
