"""Tests for the HPO trial result dataclass."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from drevalpy.types.results.trial import TrialResult


def _trial(**overrides: Any) -> TrialResult:
    kwargs: dict[str, Any] = {
        "hyperparameters": {"alpha": 0.5},
        "metrics": {"MSE": 0.25, "Pearson": 0.75},
        "optimization_metric": "MSE",
        "predictions": np.array([1.0, 2.0, 3.0]),
    }
    kwargs.update(overrides)
    return TrialResult(**kwargs)


def test_score_returns_value_of_optimization_metric() -> None:
    trial = _trial(optimization_metric="Pearson")

    assert trial.score == 0.75


def test_score_falls_back_to_nan_when_metric_missing() -> None:
    trial = _trial(optimization_metric="Kendall")

    assert math.isnan(trial.score)


def test_score_is_nan_for_empty_metrics() -> None:
    trial = _trial(metrics={})

    assert math.isnan(trial.score)


def test_repr_lists_every_hyperparameter() -> None:
    trial = _trial(hyperparameters={"alpha": 0.5, "l1_ratio": 0.1})

    text = repr(trial)

    assert "alpha: 0.5" in text
    assert "l1_ratio: 0.1" in text


def test_repr_marks_only_the_optimization_metric() -> None:
    trial = _trial(optimization_metric="MSE")

    lines = repr(trial).splitlines()

    assert "        MSE: 0.2500 *" in lines
    assert "        Pearson: 0.7500" in lines


def test_repr_starts_with_the_type_name() -> None:
    assert repr(_trial()).startswith("TrialResult")
