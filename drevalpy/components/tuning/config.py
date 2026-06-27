"""Ray Tune search configuration for DRP experiment hyperparameter tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
        from drevalpy.evaluation import get_mode

        return cls(n_trials=n_trials, metric=metric, mode=get_mode(metric), **kwargs)
