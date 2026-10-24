"""Tests for the public surface of the results package."""

from __future__ import annotations

from drevalpy.types import results
from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.types.results.failure import ExperimentFailedError, RunFailure
from drevalpy.types.results.model import ModelResult
from drevalpy.types.results.run import RunResult
from drevalpy.types.results.trial import TrialResult


def test_all_lists_every_result_type() -> None:
    assert sorted(results.__all__) == [
        "ExperimentFailedError",
        "ExperimentResult",
        "ModelResult",
        "RunFailure",
        "RunResult",
        "TrialResult",
    ]


def test_re_exports_are_the_defining_classes() -> None:
    assert results.ExperimentResult is ExperimentResult
    assert results.ExperimentFailedError is ExperimentFailedError
    assert results.ModelResult is ModelResult
    assert results.RunFailure is RunFailure
    assert results.RunResult is RunResult
    assert results.TrialResult is TrialResult
