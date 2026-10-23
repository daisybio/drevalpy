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


class TestFromException:
    def test_records_the_type_and_message(self) -> None:
        failure = RunFailure.from_exception(KeyError("gene"), model_name="A", fold_index=2)

        assert failure == _failure(fold_index=2, error_type="KeyError", message="'gene'")

    def test_records_the_randomization_and_robustness_trial(self) -> None:
        failure = RunFailure.from_exception(
            RuntimeError("x"),
            model_name="A",
            fold_index=0,
            randomization=("SVRC", "gene_expression"),
            robustness_trial=3,
        )

        assert failure.randomization == ("SVRC", "gene_expression")
        assert failure.robustness_trial == 3

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            _failure().message = "other"  # type: ignore[misc]


class TestDict:
    @pytest.mark.parametrize(
        "failure",
        [_failure(), _failure(randomization=("SVRC", "gene_expression"), robustness_trial=0)],
        ids=["plain", "stress-tests"],
    )
    def test_round_trips_through_json(self, failure: RunFailure) -> None:
        data = json.loads(json.dumps(failure.to_dict()))

        assert RunFailure.from_dict(data) == failure

    def test_to_dict_is_json_safe_with_a_randomization(self) -> None:
        data = _failure(randomization=("SVRC", "gene_expression")).to_dict()

        assert data["randomization"] == ["SVRC", "gene_expression"]

    def test_from_dict_tolerates_absent_stress_test_fields(self) -> None:
        data = {"model_name": "A", "fold_index": 1, "error_type": "ValueError", "message": "boom"}

        assert RunFailure.from_dict(data) == _failure()


class TestStr:
    def test_names_the_model_fold_and_error(self) -> None:
        assert str(_failure()) == "A fold 1: ValueError: boom"

    def test_mentions_the_stress_tests(self) -> None:
        text = str(_failure(randomization=("SVRC", "gene_expression"), robustness_trial=2))

        assert "randomization SVRC (gene_expression)" in text
        assert "robustness trial 2" in text


class TestExperimentFailedError:
    def test_carries_the_failures_and_is_a_runtime_error(self) -> None:
        failures = [_failure(fold_index=i) for i in range(2)]

        error = ExperimentFailedError(failures)

        assert isinstance(error, RuntimeError)
        assert isinstance(error, DrEvalError)
        assert error.failures == tuple(failures)

    def test_message_counts_the_failures_and_lists_every_one_indented(self) -> None:
        failures = [_failure(fold_index=i) for i in range(8)]

        lines = str(ExperimentFailedError(failures)).splitlines()

        assert lines[0] == "8 run(s) failed:"
        assert lines[1:] == [f"  {failure}" for failure in failures]
