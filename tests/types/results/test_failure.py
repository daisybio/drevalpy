"""Records of failed runs and the error raised when every run failed."""

from __future__ import annotations

import dataclasses
import json

import pytest
from upath import UPath

from drevalpy.errors import DrEvalError
from drevalpy.types.results._identity import RunIdentity
from drevalpy.types.results.failure import ExperimentFailedError, RunFailure, read_failures, write_failures
from tests.synthetic import make_run_failure


def test_from_exception_records_the_identity_type_and_message() -> None:
    identity = RunIdentity("A", "SyntheticDataset", "LPO", 2, randomization=("SVRC", "gene_expression"))

    failure = RunFailure.from_exception(KeyError("gene"), identity)

    assert failure == RunFailure(identity, "KeyError", "'gene'")
    with pytest.raises(dataclasses.FrozenInstanceError):
        failure.message = "other"  # type: ignore[misc]


def test_failures_round_trip_through_failures_json(tmp_path) -> None:
    directory = UPath(tmp_path)
    failures = [
        make_run_failure("A", 1),
        make_run_failure("B", 0, ("SVRC", "gene_expression"), 2, "KeyError", "'gene'"),
    ]

    write_failures(directory, failures)

    stored = json.loads((directory / "failures.json").read_text(encoding="utf-8"))
    assert stored[1]["identity"]["randomization"] == ["SVRC", "gene_expression"]
    assert read_failures(directory) == failures


def test_no_file_is_written_without_failures_and_none_reads_as_empty(tmp_path) -> None:
    write_failures(UPath(tmp_path), [])

    assert not (UPath(tmp_path) / "failures.json").exists()
    assert read_failures(UPath(tmp_path)) == []


def test_str_names_model_fold_error_and_stress_tests() -> None:
    assert str(make_run_failure("A", 1, error_type="ValueError")) == "A fold 1: ValueError: boom"
    text = str(make_run_failure(randomization=("SVRC", "gene_expression"), robustness_repeat=2))
    assert "randomization SVRC (gene_expression)" in text
    assert "robustness repeat 2" in text


def test_experiment_failed_error_carries_and_lists_the_failures() -> None:
    failures = [make_run_failure(fold_index=i) for i in range(8)]
    error = ExperimentFailedError(failures)

    assert isinstance(error, RuntimeError)
    assert isinstance(error, DrEvalError)
    assert error.failures == tuple(failures)
    lines = str(error).splitlines()
    assert lines[0] == "8 run(s) failed:"
    assert lines[1:] == [f"  {failure}" for failure in failures]
