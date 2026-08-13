"""Visualization example: a residual histogram for one model.

``Visualization`` declares four abstract methods -- ``compute``, ``to_png``,
``to_multiqc`` and ``show``. ``ImageVisualization`` implements the last three on
top of a Matplotlib figure, so a static plot only has to supply ``compute`` (which
must leave the figure in ``self._fig``) and ``_create_figure``.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from drevalpy.plugin import (
    Dataset,
    ImageVisualization,
    ModelResult,
    register_visualization,
)


@register_visualization(
    "toyResiduals",
    "Histogram of prediction residuals pooled across a model's folds.",
    result_type="ModelResult",
    requirements=frozenset(),
)
class ToyResidualHistogram(ImageVisualization):
    """Pool every fold's residuals and draw one histogram."""

    def __init__(self) -> None:
        """Create an empty visualization."""
        self._residuals = np.empty(0, dtype=np.float64)
        self._title = ""

    def compute(self, result: ModelResult, dataset: Dataset | None = None) -> None:
        """Pool the residuals and build the figure.

        ``compute`` must assign ``self._fig``; the inherited ``to_png``,
        ``to_multiqc`` and ``show`` all raise until it does.

        Args:
            result: The model whose folds are pooled.
            dataset: Unused here. It is offered so a plot can look up cell-line
                or drug metadata that the result itself does not carry.
        """
        _ = dataset
        residuals = [np.asarray(run.predictions) - np.asarray(run.ground_truth) for run in result.runs]
        pooled = np.concatenate(residuals) if residuals else np.empty(0)
        self._residuals = pooled[~np.isnan(pooled)]
        self._title = f"{result.model_name} residuals ({len(result.runs)} folds)"
        self._fig = self._create_figure()

    def _create_figure(self) -> matplotlib.figure.Figure:
        """Draw the histogram."""
        figure, axes = plt.subplots(figsize=(6, 4))
        axes.hist(self._residuals, bins=20)
        axes.set_xlabel("predicted - observed")
        axes.set_ylabel("pairs")
        axes.set_title(self._title)
        figure.tight_layout()
        return figure
