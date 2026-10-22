"""Weights & Biases tracking that costs nothing when it is switched off.

``wandb`` is imported inside the functions that use it, never at module scope: it is on
the import-cost policy's forbidden list, and a run without a project must not pay for it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager


def tracking_run(
    project: str | None,
    *,
    name: str,
    config: Mapping | None = None,
    tags: Sequence[str] = (),
) -> AbstractContextManager[None]:
    """Open a W&B run for the duration of a ``with`` block.

    Args:
        project: W&B project to log to. ``None`` makes the context manager a no-op that
            does not import ``wandb``.
        name: Name of the W&B run, such as ``"MyModel_split0_final"``.
        config: Hyperparameters and settings recorded as the run's config.
        tags: Tags attached to the run.

    Returns:
        A context manager that starts the run on entry and finishes it on exit, also when
        the block raises.
    """
    raise NotImplementedError("W1-P6")


def log_summary(metrics: Mapping[str, float]) -> None:
    """Write final metrics to the summary of the active W&B run.

    Does nothing when no run is active, so callers need not check whether tracking is on.

    Args:
        metrics: Metric name to value, such as ``{"test_RMSE": 1.2}``.
    """
    raise NotImplementedError("W1-P6")
