"""Tests for the public surface of the experiment package."""

from __future__ import annotations

from drevalpy import experiment
from drevalpy.experiment.randomization import randomization
from drevalpy.experiment.robustness import robustness
from drevalpy.types.results.run import RunResult
from drevalpy.types.results.trial import TrialResult


def test_all_lists_the_documented_surface() -> None:
    assert sorted(experiment.__all__) == ["RunResult", "TrialResult", "randomization", "robustness"]


def test_re_exports_the_experiment_helpers() -> None:
    assert experiment.randomization is randomization
    assert experiment.robustness is robustness


def test_re_exports_the_result_types() -> None:
    assert experiment.RunResult is RunResult
    assert experiment.TrialResult is TrialResult
