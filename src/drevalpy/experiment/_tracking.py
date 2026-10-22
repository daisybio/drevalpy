"""Weights & Biases tracking that costs nothing when it is switched off.

``wandb`` is imported inside the functions that use it, never at module scope: it is on
the import-cost policy's forbidden list, and a run without a project must not pay for it.
Code that only reads the active run (``log_summary``, ``log_metrics``, ``active_run``)
looks ``wandb`` up in ``sys.modules`` instead of importing it: a run can only be active
if something already imported the module.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any


@contextmanager
def tracking_run(
    project: str | None,
    *,
    name: str,
    config: Mapping[str, Any] | None = None,
    tags: Sequence[str] = (),
) -> Iterator[None]:
    """Open a W&B run for the duration of a ``with`` block.

    :param project: W&B project to log to. ``None`` makes the context manager a no-op that
        does not import ``wandb``.
    :param name: Name of the W&B run, such as ``"MyModel_split0_final"``.
    :param config: Hyperparameters and settings recorded as the run's config.
    :param tags: Tags attached to the run.
    :returns: A context manager that starts the run on entry, finishing any previous run
        first, and finishes it on exit. A run left by an exception is marked failed.
    """
    if project is None:
        yield
        return

    wandb = _import_wandb()
    if wandb.run is not None:
        wandb.finish()
    wandb.init(project=project, name=name, config=dict(config or {}), tags=list(tags))
    try:
        yield
    except BaseException:
        wandb.finish(exit_code=1)
        raise
    wandb.finish()


def require_tracking(project: str | None) -> None:
    """Fail now, rather than inside the first trial, when tracking cannot work.

    :param project: W&B project, or ``None`` when tracking is off.
    :raises ImportError: If *project* is set and ``wandb`` is not installed.
    """
    if project is not None:
        _import_wandb()


def log_summary(metrics: Mapping[str, float]) -> None:
    """Write final metrics to the summary of the active W&B run.

    Does nothing when no run is active, so callers need not check whether tracking is on.

    :param metrics: Metric name to value, such as ``{"test_RMSE": 1.2}``.
    """
    run = active_run()
    if run is not None:
        run.summary.update(dict(metrics))


def log_metrics(metrics: Mapping[str, float]) -> None:
    """Log one step of per-epoch metrics to the active W&B run.

    Does nothing when no run is active. Lightning predictors get the same curves through
    the logger attached in ``run_lightning_fit``; this is for hand-written training loops.

    :param metrics: Metric name to value, such as ``{"epoch": 3, "val_loss": 0.4}``.
    """
    if active_run() is not None:
        sys.modules["wandb"].log(dict(metrics))


def active_run() -> Any | None:
    """Return the active W&B run without importing ``wandb``.

    :returns: ``wandb.run``, or ``None`` when ``wandb`` was never imported or no run is open.
    """
    wandb = sys.modules.get("wandb")
    return getattr(wandb, "run", None)


def _import_wandb() -> Any:
    """Import ``wandb``, explaining how to install it when it is missing.

    :returns: The ``wandb`` module.
    :raises ImportError: If ``wandb`` is not installed.
    """
    try:
        import wandb
    except ImportError as exc:
        msg = "Weights & Biases tracking needs the 'wandb' package; install it with `pip install wandb`."
        raise ImportError(msg) from exc
    return wandb
