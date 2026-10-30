"""Optuna search configuration for DRP experiment hyperparameter tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from drevalpy.evaluation import Metric


@dataclass(frozen=True)
class HPOConfig:
    """Settings of one Optuna hyperparameter search: trial count, objective metric and sampler seed."""

    n_trials: int = 16
    metric: str = "RMSE"
    random_state: int = 42

    def __post_init__(self) -> None:
        """Reject an unknown metric and a negative number of trials.

        ``Metric(metric)`` raises ``ValueError`` for a name it does not know.

        :raises ValueError: If ``n_trials`` is negative.
        """
        Metric(self.metric)
        if self.n_trials < 0:
            raise ValueError(f"n_trials must be >= 0 (got {self.n_trials}); use 0 for default-only tuning")

    @property
    def direction(self) -> Literal["minimize", "maximize"]:
        """Return Optuna's study direction for :attr:`metric`."""
        return "minimize" if Metric(self.metric).direction == "min" else "maximize"
