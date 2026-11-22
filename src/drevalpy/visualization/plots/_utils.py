"""Shared utilities for visualization plot implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

    from drevalpy.types.results import ExperimentResult


def placeholder_figure(message: str, figsize: tuple[float, float]) -> Figure:
    """Draw a figure holding only *message*, for a plot with nothing to show.

    :param message: Text centred on the otherwise empty figure.
    :param figsize: Figure size in inches.
    :returns: The Matplotlib figure, built directly so it never enters pyplot's registry.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot()
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    return fig


def runs_frame(
    result: ExperimentResult,
    *,
    randomization: tuple[str, str] | Literal["all"] | None = None,
    robustness: int | Literal["all"] | None = None,
) -> pd.DataFrame:
    """Flatten every run of *result* into one row per model, setting and fold.

    Columns are ``algorithm``, ``rand_setting``, ``test_mode``, ``CV_split`` plus
    one per metric the run reports.

    :param result: Experiment result to flatten.
    :param randomization: Exact setting, ``"all"``, or ``None`` for ordinary runs.
    :param robustness: Exact repeat, ``"all"``, or ``None`` for ordinary runs.
    :returns: The per-run frame.
    """
    import pandas as pd

    rows: list[dict] = []
    for model in result.models:
        for run in model.select_runs(randomization=randomization, robustness=robustness):
            setting = "_".join(run.randomization) if run.randomization else "predictions"
            row: dict = {
                "algorithm": run.model_name,
                "rand_setting": setting,
                "test_mode": result.split_mode,
                "CV_split": run.fold_index,
            }
            row.update(run.metrics)
            rows.append(row)
    return pd.DataFrame(rows)
