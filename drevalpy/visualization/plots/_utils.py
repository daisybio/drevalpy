"""Shared utilities for visualization plot implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    import pandas as pd

    from drevalpy.types.results import ExperimentResult

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


def runs_frame(result: ExperimentResult, *, indexed: bool = False) -> pd.DataFrame:
    """Flatten every run of *result* into one row per model, setting and fold.

    Columns are ``algorithm``, ``rand_setting``, ``test_mode``, ``CV_split`` plus
    one per metric the run reports.

    :param result: Experiment result to flatten.
    :param indexed: Label each row ``<model>_<setting>_<mode>_split_<fold>``. The
        heatmap and the cross-study table group on that label; the violin plot
        wants the default positional index.
    :returns: The per-run frame.
    """
    import pandas as pd

    rows: list[dict] = []
    labels: list[str] = []
    for model in result.models:
        for run in model.runs:
            setting = f"{run.randomization[0]}_{run.randomization[1]}" if run.randomization else "predictions"
            row: dict = {
                "algorithm": run.model_name,
                "rand_setting": setting,
                "test_mode": result.split_mode,
                "CV_split": run.fold_index,
            }
            row.update(run.metrics)
            rows.append(row)
            labels.append(f"{run.model_name}_{setting}_{result.split_mode}_split_{run.fold_index}")
    if indexed:
        return pd.DataFrame(rows, index=labels)
    return pd.DataFrame(rows)
