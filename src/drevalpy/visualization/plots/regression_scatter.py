"""Regression scatter: predicted against observed response, as a hexbin density.

At production scale one model contributes ~231k predictions across its folds,
which is two orders of magnitude past MultiQC's ``plots_flat_numseries`` cutoff -
the interactive scatter was already being flattened to a static image, and the
per-point payload bought nothing. A log-scaled hexbin shows the same cloud, plus
the density structure that overplotting hides, from two float arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drevalpy.registry.visualization import register
from drevalpy.types.results import ModelResult
from drevalpy.visualization.base import Plot, Section, embedded_png_html

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Hexagons per axis. 40 keeps the bins visible at report width without
#: degenerating into one-observation cells on a 20k-point cloud.
_GRID_SIZE = 40


def _pooled_predictions(result: ModelResult) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the finite (ground truth, prediction) pairs over all folds.

    :param result: Model result whose non-randomized runs to pool.
    :returns: Two aligned ``float64`` arrays, empty when nothing is scorable.
    """
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for run in result.select_runs():
        truth = np.asarray(run.ground_truth, dtype=np.float64)
        prediction = np.asarray(run.predictions, dtype=np.float64)
        keep = np.isfinite(truth) & np.isfinite(prediction)
        truths.append(truth[keep])
        predictions.append(prediction[keep])
    if not truths:
        return np.empty(0), np.empty(0)
    return np.concatenate(truths), np.concatenate(predictions)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of two aligned arrays, NaN where undefined.

    :param x: First variable.
    :param y: Second variable, aligned with ``x``.
    :returns: The correlation, or NaN for fewer than two points or zero variance
        on either side.
    """
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


@register("regression_scatter", "Density of predicted vs. observed drug response values")
class RegressionScatterPlot(Plot):
    """Predicted vs. ground-truth hexbin density for a single model."""

    result_type = ModelResult

    def figure(self, result: ModelResult) -> Figure:
        """Draw the pooled predictions as a hexbin with an identity line and a fit annotation.

        Built on :class:`matplotlib.figure.Figure` directly rather than through
        ``pyplot``, so the figure never enters pyplot's global registry and is
        released with its last reference.

        :param result: Model result containing predictions and ground truth.
        :returns: The rendered figure.
        """
        from matplotlib.figure import Figure

        ground_truth, predictions = _pooled_predictions(result)
        fig = Figure(figsize=(7, 6.5), layout="constrained")
        ax = fig.add_subplot()

        if ground_truth.size == 0:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            ax.set_axis_off()
            return fig

        low = float(min(ground_truth.min(), predictions.min()))
        high = float(max(ground_truth.max(), predictions.max()))
        if low == high:
            low, high = low - 0.5, high + 0.5

        hexes = ax.hexbin(
            ground_truth,
            predictions,
            gridsize=_GRID_SIZE,
            bins="log",
            cmap="viridis",
            mincnt=1,
            extent=(low, high, low, high),
        )
        fig.colorbar(hexes, ax=ax, label="Predictions per bin (log scale)")
        ax.plot([low, high], [low, high], linestyle="--", linewidth=1, color="#d62728")

        pcc = _pearson(ground_truth, predictions)
        ax.set_title(f"{result.model_name}: predicted vs. observed response")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.text(
            0.03,
            0.97,
            f"n = {ground_truth.size:,}\nPearson = {pcc:.3f}\nR² = {pcc**2:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        return fig

    def sections(self, result: ModelResult) -> list[Section]:
        """Embed the figure in a Section anchored on the model name.

        The default anchors on the plot name alone, which would collide across the
        one module per model that the report adds.

        :param result: Model result containing predictions and ground truth.
        :returns: A single-element list.
        """
        return [
            Section(
                name=f"Regression density: {result.model_name}",
                anchor=f"dreval_scatter_{result.model_name}",
                description=(
                    f"Predicted vs. ground-truth values for {result.model_name} "
                    f"across {result.n_folds} fold(s), binned by density."
                ),
                content=embedded_png_html(self.figure(result)),
            )
        ]
