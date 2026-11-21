"""Plot example: a residual histogram for one model.

A plot is stateless. ``figure(result)`` draws the result it is given, and the
inherited ``sections(result)`` embeds that figure in the report as one PNG section.
``result_type = ModelResult`` makes the report draw it once per model; the default,
``ExperimentResult``, draws it once for the whole experiment.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from drevalpy.plugin import (
    ModelResult,
    Plot,
    register_visualization,
)


@register_visualization("toyResiduals", "Histogram of prediction residuals pooled across a model's folds.")
class ToyResidualHistogram(Plot):
    """Pool every fold's residuals and draw one histogram."""

    result_type = ModelResult

    def figure(self, result: ModelResult) -> matplotlib.figure.Figure:
        """Draw the histogram of *result*'s residuals.

        :param result: The model whose folds are pooled.
        """
        residuals = [np.asarray(run.predictions) - np.asarray(run.ground_truth) for run in result.runs]
        pooled = np.concatenate(residuals) if residuals else np.empty(0)
        figure, axes = plt.subplots(figsize=(6, 4))
        axes.hist(pooled[~np.isnan(pooled)], bins=20)
        axes.set_xlabel("predicted - observed")
        axes.set_ylabel("pairs")
        axes.set_title(f"{result.model_name} residuals ({len(result.runs)} folds)")
        figure.tight_layout()
        return figure
