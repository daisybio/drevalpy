"""Tests for :mod:`drevalpy.cli.aggregate`, run end to end against real ``RunResult`` serialization."""

from __future__ import annotations

import json

import pytest
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.types.results import ExperimentResult
from tests.cli._helpers import HELP_ENV, make_runner, plain
from tests.synthetic import DEFAULT_DATASET_NAME, make_run_result

runner = make_runner()


@pytest.fixture()
def run_files(tmp_path: UPath) -> list[UPath]:
    paths: list[UPath] = []
    for model_name in ("ElasticNet", "RandomForest"):
        for fold_index in range(2):
            path = tmp_path / f"{model_name}_fold_{fold_index}.npz"
            make_run_result(model_name=model_name, fold_index=fold_index).save(path)
            paths.append(path)
    return paths


def _invoke(run_files: list[UPath], out_dir: UPath, *extra: str):
    return runner.invoke(app, ["aggregate", *[str(p) for p in run_files], "--output-dir", str(out_dir), *extra])


def test_missing_results_is_a_usage_error() -> None:
    assert runner.invoke(app, ["aggregate"], env=HELP_ENV).exit_code == 2


def test_nonexistent_result_file_is_reported_and_creates_no_output(tmp_path: UPath) -> None:
    result = _invoke([tmp_path / "absent.npz"], tmp_path / "out")

    assert result.exit_code == 1
    assert plain(result.output).startswith("Error: ")
    assert not (tmp_path / "out").exists()


def test_aggregates_runs_into_a_reloadable_experiment(run_files: list[UPath], tmp_path: UPath) -> None:
    out_dir = tmp_path / "nested" / "experiment"

    result = _invoke(run_files, out_dir)

    assert result.exit_code == 0, result.output
    text = plain(result.output)
    assert f"Aggregated 4 runs into ExperimentResult at {out_dir}" in text
    assert "ExperimentResult" in text
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert sorted(meta["models"]) == ["ElasticNet", "RandomForest"]
    assert meta["dataset_name"] == DEFAULT_DATASET_NAME
    experiment = ExperimentResult.load(out_dir)
    assert (experiment.n_models, experiment.max_folds) == (2, 2)


def test_accepts_a_single_run(tmp_path: UPath) -> None:
    path = tmp_path / "one.npz"
    make_run_result().save(path)

    assert _invoke([path], tmp_path / "out").exit_code == 0


def test_mismatched_dataset_names_are_rejected(tmp_path: UPath) -> None:
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    make_run_result(dataset_name="DatasetA").save(first)
    make_run_result(dataset_name="DatasetB").save(second)

    result = _invoke([first, second], tmp_path / "out")

    assert result.exit_code == 1
    assert "Error: All RunResults must share the same dataset_name" in plain(result.output)


@pytest.mark.parametrize("option", ["drug", "sample", "DRUG"])
def test_group_by_prints_the_grouped_summary(run_files: list[UPath], tmp_path: UPath, option: str) -> None:
    result = _invoke(run_files, tmp_path / "out", "--group-by", option)

    assert result.exit_code == 0, result.output
    text = plain(result.output)
    assert f"Metrics computed within each {option.lower()}" in text
    assert "ElasticNet" in text.split("Metrics computed within")[1]


def test_group_by_is_silent_by_default_and_rejects_unknown_groupings(run_files: list[UPath], tmp_path: UPath) -> None:
    assert "Metrics computed within" not in plain(_invoke(run_files, tmp_path / "out").output)
    assert _invoke(run_files, tmp_path / "out2", "--group-by", "tissue").exit_code == 2
