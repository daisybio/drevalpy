"""Tests for :mod:`drevalpy.cli.report`, the ``drevalpy report`` command.

``create_report`` drives MultiQC, which is far too heavy for a CLI plumbing
test, so the worker is patched and the assertions cover argument forwarding, the
optional dataset enrichment and the echoed confirmation. Report rendering itself
belongs to the ``tests/visualization`` mirror.
"""

from __future__ import annotations

import pytest
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.types.results import ExperimentResult
from tests.cli._helpers import HELP_ENV, FakeDataset, Recorder, make_runner, patch_worker, plain
from tests.synthetic import make_experiment_result

runner = make_runner()


@pytest.fixture()
def experiment_dir(tmp_path: UPath) -> UPath:
    """A real saved ``ExperimentResult`` tree, so ``load`` stays unpatched."""
    path = tmp_path / "experiment"
    make_experiment_result(n_models=2, n_folds=2).save(path)
    return path


@pytest.fixture()
def worker(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Patch :func:`drevalpy.visualization.report.create_report`.

    Args:
        monkeypatch: Fixture used to replace the source-module worker.

    Returns:
        Recorder standing in for the report writer.
    """
    recorder = Recorder()
    patch_worker(monkeypatch, "drevalpy.visualization.report", "create_report", recorder)
    return recorder


class TestArguments:
    """The experiment directory is the one required argument."""

    def test_missing_experiment_dir_is_a_usage_error(self, worker: Recorder) -> None:
        result = runner.invoke(app, ["report"], env=HELP_ENV)

        assert result.exit_code == 2

    def test_nonexistent_experiment_dir_fails(self, worker: Recorder, tmp_path: UPath) -> None:
        result = runner.invoke(app, ["report", str(tmp_path / "absent")])

        assert result.exit_code != 0

    def test_nonexistent_experiment_dir_does_not_reach_the_worker(self, worker: Recorder, tmp_path: UPath) -> None:
        runner.invoke(app, ["report", str(tmp_path / "absent")])

        assert worker.call_count == 0


class TestForwarding:
    """Options map onto :func:`create_report`'s signature."""

    def test_exits_cleanly(self, worker: Recorder, experiment_dir: UPath) -> None:
        result = runner.invoke(app, ["report", str(experiment_dir)])

        assert result.exit_code == 0, result.output

    def test_passes_the_loaded_experiment_and_output_dir_positionally(
        self, worker: Recorder, experiment_dir: UPath
    ) -> None:
        runner.invoke(app, ["report", str(experiment_dir), "--output-dir", "my_report"])

        experiment, output_dir = worker.args
        assert isinstance(experiment, ExperimentResult)
        assert output_dir == "my_report"

    def test_default_output_dir_and_title(self, worker: Recorder, experiment_dir: UPath) -> None:
        runner.invoke(app, ["report", str(experiment_dir)])

        assert worker.args[1] == "report"
        assert worker.kwargs["title"] == "Drug Response Evaluation"

    def test_title_short_option(self, worker: Recorder, experiment_dir: UPath) -> None:
        runner.invoke(app, ["report", str(experiment_dir), "-t", "My Title"])

        assert worker.kwargs["title"] == "My Title"

    def test_reference_model_defaults_to_none(self, worker: Recorder, experiment_dir: UPath) -> None:
        runner.invoke(app, ["report", str(experiment_dir)])

        assert worker.kwargs["reference_model"] is None

    def test_reference_model_short_option(self, worker: Recorder, experiment_dir: UPath) -> None:
        runner.invoke(app, ["report", str(experiment_dir), "-r", "NaiveMeanEffectsPredictor"])

        assert worker.kwargs["reference_model"] == "NaiveMeanEffectsPredictor"

    def test_echoes_the_output_directory(self, worker: Recorder, experiment_dir: UPath) -> None:
        result = runner.invoke(app, ["report", str(experiment_dir), "-o", "out_here"])

        assert "Report generated at out_here" in plain(result.output)


class TestDatasetEnrichment:
    """``--dataset`` is optional and only then is ``Dataset.load`` reached."""

    def test_dataset_is_none_by_default(self, worker: Recorder, experiment_dir: UPath) -> None:
        runner.invoke(app, ["report", str(experiment_dir)])

        assert worker.kwargs["dataset"] is None

    def test_dataset_is_loaded_and_forwarded(
        self, worker: Recorder, experiment_dir: UPath, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
    ) -> None:
        dataset = FakeDataset()
        monkeypatch.setattr("drevalpy.types.data.dataset.Dataset.load", classmethod(lambda cls, path: dataset))

        runner.invoke(app, ["report", str(experiment_dir), "-d", str(tmp_path / "ds.h5mu")])

        assert worker.kwargs["dataset"] is dataset

    def test_dataset_path_is_passed_through_to_the_loader(
        self, worker: Recorder, experiment_dir: UPath, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath
    ) -> None:
        seen: list[str] = []
        monkeypatch.setattr(
            "drevalpy.types.data.dataset.Dataset.load",
            classmethod(lambda cls, path: seen.append(path) or FakeDataset()),
        )
        dataset_path = tmp_path / "ds.h5mu"

        runner.invoke(app, ["report", str(experiment_dir), "--dataset", str(dataset_path)])

        assert seen == [str(dataset_path)]
