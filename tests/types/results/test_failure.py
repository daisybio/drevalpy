"""Tests for the record of a failed run."""

from __future__ import annotations

import dataclasses
import json

import pytest

from drevalpy.errors import DrEvalError
from drevalpy.types.results.failure import ExperimentFailedError, RunFailure


def _failure(**overrides) -> RunFailure:
    fields = {
        "model_name": "A",
        "fold_index": 1,
        "randomization": None,
        "robustness_trial": None,
        "error_type": "ValueError",
        "message": "boom",
    }
    return RunFailure(**{**fields, **overrides})


def test_from_exception_records_type_message_and_stress_tests() -> None:
    failure = RunFailure.from_exception(KeyError("gene"), model_name="A", fold_index=2)
    assert failure == _failure(fold_index=2, error_type="KeyError", message="'gene'")

    failure = RunFailure.from_exception(
        RuntimeError("x"), model_name="A", fold_index=0, randomization=("SVRC", "gene_expression"), robustness_trial=3
    )
    assert (failure.randomization, failure.robustness_trial) == (("SVRC", "gene_expression"), 3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        failure.message = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "failure",
    [_failure(), _failure(randomization=("SVRC", "gene_expression"), robustness_trial=0)],
    ids=["plain", "stress"],
)
def test_dict_round_trips_through_json(failure: RunFailure) -> None:
    data = json.loads(json.dumps(failure.to_dict()))
    assert RunFailure.from_dict(data) == failure


def test_dict_edge_cases() -> None:
    assert _failure(randomization=("SVRC", "gene_expression")).to_dict()["randomization"] == ["SVRC", "gene_expression"]
    data = {"model_name": "A", "fold_index": 1, "error_type": "ValueError", "message": "boom"}
    assert RunFailure.from_dict(data) == _failure()


def test_str_names_model_fold_error_and_stress_tests() -> None:
    assert str(_failure()) == "A fold 1: ValueError: boom"
    text = str(_failure(randomization=("SVRC", "gene_expression"), robustness_trial=2))
    assert "randomization SVRC (gene_expression)" in text
    assert "robustness trial 2" in text


def test_experiment_failed_error_carries_and_lists_the_failures() -> None:
    failures = [_failure(fold_index=i) for i in range(8)]
    error = ExperimentFailedError(failures)

    assert isinstance(error, RuntimeError)
    assert isinstance(error, DrEvalError)
    assert error.failures == tuple(failures)
    lines = str(error).splitlines()
    assert lines[0] == "8 run(s) failed:"
    assert lines[1:] == [f"  {failure}" for failure in failures]
