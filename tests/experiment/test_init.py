"""Tests for init."""

from __future__ import annotations

from drevalpy import experiment
from drevalpy.types.results.run import RunResult
from drevalpy.types.results.trial import TrialResult


def test_surface() -> None:
    assert sorted(experiment.__all__) == ["randomization", "robustness"]
    assert experiment.RunResult is RunResult
    assert experiment.TrialResult is TrialResult
