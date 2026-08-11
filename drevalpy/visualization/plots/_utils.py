"""Shared utilities for visualization plot implementations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

MODEL_COLORS: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def model_color_palette(model_names: list[str]) -> dict[str, str]:
    """Assign a distinct color to each model name.

    Colors cycle through MODEL_COLORS if there are more models than colors.

    :param model_names: List of model names.
    :returns: Mapping from model name to hex color string.
    """
    return {name: MODEL_COLORS[i % len(MODEL_COLORS)] for i, name in enumerate(model_names)}


def compute_ssmd(values_a: ArrayLike, values_b: ArrayLike) -> float:
    """Compute the Strictly Standardized Mean Difference between two arrays.

    SSMD = (mean_a - mean_b) / sqrt(var_a + var_b)

    :param values_a: Metric values for model A (across CV folds).
    :param values_b: Metric values for model B (across CV folds).
    :returns: SSMD value, or NaN if the denominator is zero.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    denom = var_a + var_b
    if denom <= 0:
        return float("nan")
    return float((mu_a - mu_b) / np.sqrt(denom))
