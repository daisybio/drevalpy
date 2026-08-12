"""Tests for :mod:`drevalpy.cli.aggregate`, the ``drevalpy aggregate`` command.

Nothing here is patched: ``RunResult`` .npz files are cheap to write, so the
command is exercised end to end against real serialization.
"""

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
    """Two folds each of two models, written as RunResult .npz files."""
    paths: list[UPath] = []
    for model_name in ("ElasticNet", "RandomForest"):
        for fold_index in range(2):
            path = tmp_path / f"{model_name}_fold_{fold_index}.npz"
            make_run_result(model_name=model_name, fold_index=fold_index).save(path)
            paths.append(path)
    return paths


def _invoke(run_files: list[UPath], out_dir: UPath):
    return runner.invoke(app, ["aggregate", *[str(p) for p in run_files], "--output-dir", str(out_dir)])


class TestArguments:
    """At least one RunResult path is required."""

    def test_missing_results_is_a_usage_error(self) -> None:
        result = runner.invoke(app, ["aggregate"], env=HELP_ENV)

        assert result.exit_code == 2

    def test_nonexistent_result_file_fails(self, tmp_path: UPath) -> None:
        result = _invoke([tmp_path / "absent.npz"], tmp_path / "out")

        assert result.exit_code != 0


class TestAggregation:
    """The command groups the given runs into one saved ExperimentResult."""

    def test_exits_cleanly(self, run_files: list[UPath], tmp_path: UPath) -> None:
        result = _invoke(run_files, tmp_path / "out")

        assert result.exit_code == 0, result.output

    def test_creates_the_output_directory_including_parents(self, run_files: list[UPath], tmp_path: UPath) -> None:
        out_dir = tmp_path / "nested" / "experiment"
        _invoke(run_files, out_dir)

        assert out_dir.is_dir()

    def test_writes_experiment_metadata(self, run_files: list[UPath], tmp_path: UPath) -> None:
        out_dir = tmp_path / "out"
        _invoke(run_files, out_dir)

        meta = json.loads((out_dir / "metadata.json").read_text())
        assert sorted(meta["models"]) == ["ElasticNet", "RandomForest"]

    def test_records_the_shared_dataset_name(self, run_files: list[UPath], tmp_path: UPath) -> None:
        out_dir = tmp_path / "out"
        _invoke(run_files, out_dir)

        meta = json.loads((out_dir / "metadata.json").read_text())
        assert meta["dataset_name"] == DEFAULT_DATASET_NAME

    def test_result_is_reloadable(self, run_files: list[UPath], tmp_path: UPath) -> None:
        out_dir = tmp_path / "out"
        _invoke(run_files, out_dir)

        experiment = ExperimentResult.load(out_dir)
        assert experiment.n_models == 2
        assert experiment.max_folds == 2

    def test_echoes_the_run_count_and_destination(self, run_files: list[UPath], tmp_path: UPath) -> None:
        out_dir = tmp_path / "out"
        result = _invoke(run_files, out_dir)

        assert f"Aggregated 4 runs into ExperimentResult at {out_dir}" in plain(result.output)

    def test_echoes_the_experiment_repr(self, run_files: list[UPath], tmp_path: UPath) -> None:
        result = _invoke(run_files, tmp_path / "out")

        assert "ExperimentResult" in plain(result.output)

    def test_accepts_a_single_run(self, tmp_path: UPath) -> None:
        path = tmp_path / "one.npz"
        make_run_result().save(path)

        result = _invoke([path], tmp_path / "out")

        assert result.exit_code == 0, result.output


class TestValidation:
    """``ExperimentResult``'s own guards surface as a nonzero exit."""

    def test_mismatched_dataset_names_are_rejected(self, tmp_path: UPath) -> None:
        first = tmp_path / "a.npz"
        second = tmp_path / "b.npz"
        make_run_result(dataset_name="DatasetA").save(first)
        make_run_result(dataset_name="DatasetB").save(second)

        result = _invoke([first, second], tmp_path / "out")

        assert isinstance(result.exception, ValueError)
