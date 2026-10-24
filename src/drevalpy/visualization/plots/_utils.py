"""Shared utilities for visualization plot implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    import pandas as pd

    from drevalpy.types.results import ExperimentResult


def compute_ssmd(values_a: ArrayLike, values_b: ArrayLike) -> float:
    """Compute the Strictly Standardized Mean Difference between two arrays.

    SSMD = (mean_a - mean_b) / sqrt(var_a + var_b)

    :param values_a: Metric values for model A (across CV folds).
    :param values_b: Metric values for model B (across CV folds).
    :returns: SSMD value, or NaN if either array has fewer than two values or the denominator is zero.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    denom = var_a + var_b
    if denom <= 0:
        return float("nan")
    return float((mu_a - mu_b) / np.sqrt(denom))


def runs_frame(
    result: ExperimentResult,
    *,
    indexed: bool = False,
    randomization: tuple[str, str] | Literal["all"] | None = None,
    robustness: int | Literal["all"] | None = None,
) -> pd.DataFrame:
    """Flatten every run of *result* into one row per model, setting and fold.

    Columns are ``algorithm``, ``rand_setting``, ``test_mode``, ``CV_split`` plus
    one per metric the run reports.

    :param result: Experiment result to flatten.
    :param indexed: Label each row ``<model>_<setting>_<mode>_split_<fold>``. The
        heatmap groups on that label; the violin plot
        wants the default positional index.
    :param randomization: Exact setting, ``"all"``, or ``None`` for ordinary runs.
    :param robustness: Exact trial, ``"all"``, or ``None`` for ordinary runs.
    :returns: The per-run frame.
    """
    import pandas as pd

    rows: list[dict] = []
    labels: list[str] = []
    for model in result.models:
        for run in model.select_runs(randomization=randomization, robustness=robustness):
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
