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
from drevalpy.visualization.base import ImageVisualization, Section, embedded_png_html

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from drevalpy.types.results import ModelResult

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
    for run in result.runs:
        if run.randomization is not None:
            continue
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


@register(
    "regression_scatter",
    "Density of predicted vs. observed drug response values",
    result_type="ModelResult",
)
class RegressionScatterVisualization(ImageVisualization):
    """Predicted vs. ground-truth hexbin density for a single model."""

    def __init__(self) -> None:
        """Initialize with empty state."""
        self._fig: Figure | None = None
        self._result: ModelResult | None = None
        self._ground_truth: np.ndarray = np.empty(0)
        self._predictions: np.ndarray = np.empty(0)

    def compute(self, result: ModelResult, dataset=None) -> None:
        """Pool the model's predictions and render the density figure.

        :param result: Model result containing predictions and ground truth.
        :param dataset: Unused; accepted for interface compatibility.
        """
        self._result = result
        self._ground_truth, self._predictions = _pooled_predictions(result)
        self._fig = self._create_figure()

    def _create_figure(self) -> Figure:
        """Create the hexbin figure with an identity line and a fit annotation.

        Built on :class:`matplotlib.figure.Figure` directly rather than through
        ``pyplot``, so the figure never enters pyplot's global registry and is
        released with this object.

        :returns: The rendered figure.
        """
        from matplotlib.figure import Figure

        fig = Figure(figsize=(7, 6.5), layout="constrained")
        ax = fig.add_subplot()
        model_name = self._result.model_name if self._result is not None else ""

        if self._ground_truth.size == 0:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            ax.set_axis_off()
            return fig

        low = float(min(self._ground_truth.min(), self._predictions.min()))
        high = float(max(self._ground_truth.max(), self._predictions.max()))
        if low == high:
            low, high = low - 0.5, high + 0.5

        hexes = ax.hexbin(
            self._ground_truth,
            self._predictions,
            gridsize=_GRID_SIZE,
            bins="log",
            cmap="viridis",
            mincnt=1,
            extent=(low, high, low, high),
        )
        fig.colorbar(hexes, ax=ax, label="Predictions per bin (log scale)")
        ax.plot([low, high], [low, high], linestyle="--", linewidth=1, color="#d62728")

        pcc = _pearson(self._ground_truth, self._predictions)
        ax.set_title(f"{model_name}: predicted vs. observed response")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.text(
            0.03,
            0.97,
            f"n = {self._ground_truth.size:,}\nPearson = {pcc:.3f}\nR² = {pcc**2:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        return fig

    def to_multiqc(self) -> list[Section]:
        """Embed the figure in a Section anchored on the model name.

        The base implementation anchors on ``registry_name`` alone, which would
        collide across the one module per model that the report adds.

        :returns: A single-element list.
        :raises RuntimeError: If called before ``compute()``.
        """
        if self._fig is None or self._result is None:
            raise RuntimeError("Call compute() before to_multiqc()")
        return [
            Section(
                name=f"Regression density: {self._result.model_name}",
                anchor=f"dreval_scatter_{self._result.model_name}",
                description=(
                    f"Predicted vs. ground-truth values for {self._result.model_name} "
                    f"across {self._result.n_folds} fold(s), binned by density."
                ),
                content=embedded_png_html(self._fig),
            )
        ]
