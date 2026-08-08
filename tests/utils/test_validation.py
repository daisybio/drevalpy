"""Tests for drevalpy.utils.validation check_arguments helpers."""

from __future__ import annotations

import pathlib

import pytest

from drevalpy.utils.validation import (
    check_arguments,
    validate_measure_and_metrics,
    validate_test_modes,
)


class _MinimalArgs:
    models = ["ElasticNet"]
    baselines = None
    test_mode = ["LPO"]
    dataset_name = "GDSC1"
    no_refitting = True
    curve_curator_cores = 1
    cross_study_datasets: list[str] = []
    n_cv_splits = 2
    custom_splitter_path = None
    custom_split_name = None
    randomization_mode = ["None"]
    randomization_type = "permutation"
    n_trials_robustness = 0
    response_transformation = "None"
    measure = "LN_IC50"
    optim_metric = "Pearson"


def test_validate_test_modes_rejects_unknown() -> None:
    args = _MinimalArgs()
    args.test_mode = ["INVALID"]
    with pytest.raises(AssertionError, match="Invalid test mode"):
        validate_test_modes(args)


def test_validate_measure_rejects_unknown() -> None:
    args = _MinimalArgs()
    args.measure = "not_a_measure"
    with pytest.raises(ValueError, match="allowed drug response measures"):
        validate_measure_and_metrics(args)


def test_check_arguments_accepts_builtin_dataset(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DREVALPY_CACHE_DIR", str(tmp_path))
    args = _MinimalArgs()
    check_arguments(args)
