"""Optuna search configuration for DRP experiment hyperparameter tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from drevalpy.evaluation import AVAILABLE_METRICS, get_mode


def validate_hpo_metric(metric: str) -> None:
    """Raise ``ValueError`` when *metric* is not a supported HPO objective.

    :param metric: Evaluation metric name, for example ``"RMSE"``.
    :raises ValueError: If *metric* is not registered in ``AVAILABLE_METRICS``.
    """
    if metric not in AVAILABLE_METRICS:
        msg = f"Invalid HPO metric {metric!r}. Choose from {list(AVAILABLE_METRICS.keys())}"
        raise ValueError(msg)


@dataclass
class HPOConfig:
    """Configuration for Optuna hyperparameter search in drevalpy experiments."""

    n_trials: int = 16
    metric: str = "RMSE"
    mode: str = "min"
    random_state: int = 42

    @classmethod
    def from_metric(cls, metric: str, *, n_trials: int = 16, **kwargs: Any) -> HPOConfig:
        """Build an HPO config with ``mode`` inferred from the evaluation metric.

        :param metric: Evaluation metric name used as the Optuna objective.
        :param n_trials: Number of search trials; ``0`` selects defaults only.
        :param kwargs: Additional ``HPOConfig`` field overrides.
        :returns: Configured ``HPOConfig`` instance.
        :raises ValueError: If *metric* is invalid or *n_trials* is negative.
        """
        validate_hpo_metric(metric)
        if n_trials < 0:
            msg = f"n_trials must be >= 0 (got {n_trials}); use 0 for default-only tuning"
            raise ValueError(msg)
        return cls(n_trials=n_trials, metric=metric, mode=get_mode(metric), **kwargs)


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
