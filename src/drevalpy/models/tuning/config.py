"""Optuna search configuration for DRP experiment hyperparameter tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from drevalpy.evaluation import Metric


@dataclass
class HPOConfig:
    """Configuration for Optuna hyperparameter search in drevalpy experiments."""

    n_trials: int = 16
    metric: str = "RMSE"
    random_state: int = 42

    def __post_init__(self) -> None:
        """Reject an unknown metric: ``Metric(metric)`` raises ``ValueError`` for a name it does not know."""
        Metric(self.metric)

    @property
    def direction(self) -> Literal["minimize", "maximize"]:
        """Return Optuna's study direction for :attr:`metric`."""
        return "minimize" if Metric(self.metric).direction == "min" else "maximize"

    @classmethod
    def from_metric(cls, metric: str, *, n_trials: int = 16, **kwargs: Any) -> HPOConfig:
        r"""Build an HPO config for an evaluation metric, checking the metric before ``n_trials``.

        :param metric: Evaluation metric name used as the Optuna objective.
        :param n_trials: Number of search trials; ``0`` selects defaults only.
        :param \*\*kwargs: Additional ``HPOConfig`` field overrides.
        :returns: Configured ``HPOConfig`` instance.
        :raises ValueError: If *metric* is invalid or *n_trials* is negative.
        """
        metric = Metric(metric)
        if n_trials < 0:
            msg = f"n_trials must be >= 0 (got {n_trials}); use 0 for default-only tuning"
            raise ValueError(msg)
        return cls(n_trials=n_trials, metric=metric, **kwargs)


def build_experiment_hpo_config(
    metric: str,
    *,
    n_trials: int = 16,
    random_state: int = 42,
) -> HPOConfig:
    """Build shared Optuna settings for CV and final-model tuning.

    :param metric: Evaluation metric name used as the Optuna objective.
    :param n_trials: Number of search trials per tuning run.
    :param random_state: Random seed forwarded to the sampler.
    :returns: Configured ``HPOConfig`` instance.
    """
    return HPOConfig.from_metric(
        metric,
        n_trials=n_trials,
        random_state=random_state,
    )
