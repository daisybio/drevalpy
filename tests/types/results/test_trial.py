"""The result of one tuning trial."""

from __future__ import annotations

from typing import Any

import numpy as np

from drevalpy.types.results.trial import TrialResult


def _trial_result(**overrides: Any) -> TrialResult:
    kwargs: dict[str, Any] = {
        "hyperparameters": {"alpha": 0.5},
        "metrics": {"MSE": 0.25, "Pearson": 0.75},
        "optimization_metric": "MSE",
        "predictions": np.array([1.0, 2.0, 3.0]),
    }
    kwargs.update(overrides)
    return TrialResult(**kwargs)


def test_repr_lists_every_hyperparameter() -> None:
    trial = _trial_result(hyperparameters={"alpha": 0.5, "l1_ratio": 0.1})

    text = repr(trial)

    assert "alpha: 0.5" in text
    assert "l1_ratio: 0.1" in text


def test_repr_marks_only_the_optimization_metric() -> None:
    trial = _trial_result(optimization_metric="MSE")

    lines = repr(trial).splitlines()

    assert "        MSE: 0.2500 *" in lines
    assert "        Pearson: 0.7500" in lines


def test_repr_starts_with_the_type_name() -> None:
    assert repr(_trial_result()).startswith("TrialResult")
