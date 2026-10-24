"""Tests for :mod:`drevalpy.cli.report`; the report writer is patched, rendering is tested under tests/visualization."""

from __future__ import annotations

import pytest
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.types.results import ExperimentResult
from tests.cli._helpers import HELP_ENV, Recorder, make_runner, patch_worker, plain
from tests.synthetic import make_experiment_result

runner = make_runner()


@pytest.fixture()
def experiment_dir(tmp_path: UPath) -> UPath:
    path = tmp_path / "experiment"
    make_experiment_result(n_models=2, n_folds=2).save(path)
    return path


@pytest.fixture()
def worker(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()
    patch_worker(monkeypatch, "drevalpy.visualization.report", "create_report", recorder)
    return recorder


def test_missing_or_nonexistent_experiment_dir_never_reaches_the_worker(worker: Recorder, tmp_path: UPath) -> None:
    assert runner.invoke(app, ["report"], env=HELP_ENV).exit_code == 2
    assert runner.invoke(app, ["report", str(tmp_path / "absent")]).exit_code != 0
    assert worker.call_count == 0


def test_forwards_defaults(worker: Recorder, experiment_dir: UPath) -> None:
    result = runner.invoke(app, ["report", str(experiment_dir)])

    assert result.exit_code == 0, result.output
    experiment, output_dir = worker.args
    assert isinstance(experiment, ExperimentResult)
    assert output_dir == "report"
    assert worker.kwargs == {"title": "Drug Response Evaluation", "reference_model": None}


def test_forwards_short_options_and_echoes_the_output_directory(worker: Recorder, experiment_dir: UPath) -> None:
    result = runner.invoke(app, ["report", str(experiment_dir), "-o", "out_here", "-t", "My Title", "-r", "Naive"])

    assert worker.args[1] == "out_here"
    assert worker.kwargs == {"title": "My Title", "reference_model": "Naive"}
    assert "Report generated at out_here" in plain(result.output)


def test_the_removed_dataset_option_is_a_usage_error(worker: Recorder, experiment_dir: UPath) -> None:
    result = runner.invoke(app, ["report", str(experiment_dir), "--dataset", "ds.h5mu"])
    helped = runner.invoke(app, ["report", "--help"], env=HELP_ENV)

    assert result.exit_code == 2
    assert "--dataset" not in plain(helped.output)


def test_the_experiment_is_loaded_without_trials(
    worker: Recorder, experiment_dir: UPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []
    original = ExperimentResult.load

    def spy(directory, *, with_trials=True):
        seen.append(with_trials)
        return original(directory, with_trials=with_trials)

    monkeypatch.setattr(ExperimentResult, "load", staticmethod(spy))

    runner.invoke(app, ["report", str(experiment_dir)])

    assert seen == [False]
