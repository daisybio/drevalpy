"""Ray Tune search configuration for DRP experiment hyperparameter tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def validate_hpo_metric(metric: str) -> None:
    """Raise ``ValueError`` when *metric* is not a supported HPO objective.

    :param metric: Evaluation metric name, for example ``"RMSE"``.
    :raises ValueError: If *metric* is not registered in ``AVAILABLE_METRICS``.
    """
    from drevalpy.evaluation import AVAILABLE_METRICS

    if metric not in AVAILABLE_METRICS:
        msg = f"Invalid HPO metric {metric!r}. Choose from {list(AVAILABLE_METRICS.keys())}"
        raise ValueError(msg)


@dataclass
class HPOConfig:
    """Configuration for Ray Tune hyperparameter search in drevalpy experiments."""

    n_trials: int = 16
    metric: str = "RMSE"
    mode: str = "min"
    random_state: int = 42
    resources_per_trial: dict[str, float] = field(default_factory=lambda: {"cpu": 1})
    storage_path: str | None = None
    search_alg: str = "optuna"

    @classmethod
    def from_metric(cls, metric: str, *, n_trials: int = 16, **kwargs: Any) -> HPOConfig:
        """Build an HPO config with ``mode`` inferred from the evaluation metric.

        :param metric: Evaluation metric name used as the Ray Tune objective.
        :param n_trials: Number of search trials; ``0`` selects defaults only.
        :param kwargs: Additional ``HPOConfig`` field overrides.
        :returns: Configured ``HPOConfig`` instance.
        :raises ValueError: If *metric* is invalid or *n_trials* is negative.
        """
        from drevalpy.evaluation import get_mode

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
    resources_per_trial: dict[str, float] | None = None,
    storage_path: str | None = None,
) -> HPOConfig:
    """Build shared Ray/Optuna settings for CV and final-model tuning.

    :param metric: Evaluation metric name used as the Ray Tune objective.
    :param n_trials: Number of search trials per tuning run.
    :param random_state: Random seed forwarded to the search algorithm.
    :param resources_per_trial: Ray resource dict; defaults to one GPU when CUDA is available, otherwise one CPU.
    :param storage_path: Optional Ray Tune storage URI.
    :returns: Configured ``HPOConfig`` instance.
    """
    import torch

    resources = resources_per_trial or ({"gpu": 1} if torch.cuda.is_available() else {"cpu": 1})
    return HPOConfig.from_metric(
        metric,
        n_trials=n_trials,
        random_state=random_state,
        resources_per_trial=resources,
        storage_path=storage_path,
    )
