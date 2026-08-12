"""Tests for :mod:`drevalpy.cli.run`, the ``drevalpy run`` command."""

from __future__ import annotations

from typing import Any

import pytest
from upath import UPath

from drevalpy.cli.main import app
from tests.cli._helpers import HELP_ENV, Recorder, make_runner, patch_worker, plain

runner = make_runner()


class StubExperiment:
    """Stand-in for the ``ExperimentResult`` that ``run`` returns."""

    def __init__(self) -> None:
        """Start with an empty save log."""
        self.saved: list[str] = []

    def save(self, path: str) -> None:
        """Record the directory the command asked for.

        Args:
            path: Destination directory chosen by the command.
        """
        self.saved.append(path)

    def __repr__(self) -> str:
        """Sentinel text the command is expected to echo verbatim."""
        return "STUB-EXPERIMENT-REPR"


@pytest.fixture()
def constructed() -> list[str]:
    """Collect the model names handed to ``construct_model``."""
    return []


@pytest.fixture()
def worker(monkeypatch: pytest.MonkeyPatch, constructed: list[str]) -> Recorder:
    """Patch both lazy imports of ``run_cmd`` and hand back the ``run`` recorder.

    Args:
        monkeypatch: Fixture used to replace the source-module workers.
        constructed: List that receives every requested model name.

    Returns:
        Recorder standing in for :func:`drevalpy.run.run`.
    """

    def fake_construct_model(name: str) -> type:
        constructed.append(name)
        return type(f"Stub{name}", (), {})

    recorder = Recorder(return_value=StubExperiment())
    monkeypatch.setattr("drevalpy.models.construct_model", fake_construct_model)
    patch_worker(monkeypatch, "drevalpy.run", "run", recorder)
    return recorder


def _invoke(tmp_path: UPath, *extra: str) -> Any:
    out_dir = tmp_path / "results"
    return runner.invoke(app, ["run", "ElasticNet", "--dataset", "TOYv1", "--output-dir", str(out_dir), *extra])


class TestDefaults:
    """A minimal invocation forwards the documented default hyperparameters."""

    def test_exits_cleanly(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path)

        assert result.exit_code == 0, result.output

    def test_forwards_dataset_and_split_mode(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        assert worker.kwargs["dataset"] == "TOYv1"
        assert worker.kwargs["split_mode"] == "LPO"

    def test_forwards_hpo_defaults(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        assert worker.kwargs["hyperparameter_tuning"] is True
        assert worker.kwargs["hpo_metric"] == "RMSE"
        assert worker.kwargs["hpo_num_samples"] == 16
        assert worker.kwargs["hpo_random_state"] == 42

    def test_forwards_experiment_defaults(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        assert worker.kwargs["randomization_modes"] is None
        assert worker.kwargs["randomization_type"] == "permutation"
        assert worker.kwargs["robustness_trials"] == 0
        assert worker.kwargs["precomputed_only"] is False

    def test_passes_constructed_classes_not_names(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        assert [cls.__name__ for cls in worker.kwargs["models"]] == ["StubElasticNet"]


class TestModelArgument:
    """``models`` is variadic and every name goes through ``construct_model``."""

    def test_each_name_is_constructed(self, worker: Recorder, tmp_path: UPath, constructed: list[str]) -> None:
        out_dir = tmp_path / "results"
        runner.invoke(
            app,
            ["run", "ElasticNet", "RandomForest", "--dataset", "TOYv1", "--output-dir", str(out_dir)],
        )

        assert constructed == ["ElasticNet", "RandomForest"]

    def test_missing_models_is_a_usage_error(self, worker: Recorder) -> None:
        result = runner.invoke(app, ["run", "--dataset", "TOYv1"], env=HELP_ENV)

        assert result.exit_code == 2

    def test_missing_dataset_is_a_usage_error(self, worker: Recorder) -> None:
        result = runner.invoke(app, ["run", "ElasticNet"], env=HELP_ENV)

        assert result.exit_code == 2

    def test_nothing_runs_on_a_usage_error(self, worker: Recorder) -> None:
        runner.invoke(app, ["run", "ElasticNet"], env=HELP_ENV)

        assert worker.call_count == 0


class TestOptionForwarding:
    """Every option maps onto a keyword of :func:`drevalpy.run.run`."""

    def test_no_hpo_disables_tuning(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--no-hpo")

        assert worker.kwargs["hyperparameter_tuning"] is False

    def test_split_mode_short_option(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "-s", "LCO")

        assert worker.kwargs["split_mode"] == "LCO"

    def test_repeated_randomization_modes_become_a_list(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "-r", "SVRC", "-r", "SVCD")

        assert worker.kwargs["randomization_modes"] == ["SVRC", "SVCD"]

    def test_randomization_type(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--randomization-type", "invariant")

        assert worker.kwargs["randomization_type"] == "invariant"

    def test_hpo_tuning_options(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--hpo-metric", "Pearson", "--hpo-num-samples", "3", "--hpo-random-state", "7")

        assert worker.kwargs["hpo_metric"] == "Pearson"
        assert worker.kwargs["hpo_num_samples"] == 3
        assert worker.kwargs["hpo_random_state"] == 7

    def test_robustness_trials(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--robustness-trials", "4")

        assert worker.kwargs["robustness_trials"] == 4

    def test_precomputed_only(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--precomputed-only")

        assert worker.kwargs["precomputed_only"] is True

    def test_non_integer_hpo_samples_is_a_usage_error(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path, "--hpo-num-samples", "many")

        assert result.exit_code == 2


class TestOutput:
    """The command owns creating the output directory and reporting where it wrote."""

    def test_creates_the_output_directory_including_parents(self, worker: Recorder, tmp_path: UPath) -> None:
        out_dir = tmp_path / "nested" / "results"
        runner.invoke(app, ["run", "ElasticNet", "--dataset", "TOYv1", "--output-dir", str(out_dir)])

        assert out_dir.is_dir()

    def test_tolerates_a_pre_existing_output_directory(self, worker: Recorder, tmp_path: UPath) -> None:
        out_dir = tmp_path / "results"
        out_dir.mkdir()

        result = _invoke(tmp_path)

        assert result.exit_code == 0, result.output

    def test_saves_the_result_into_the_output_directory(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        experiment = worker.return_value
        assert experiment.saved == [str(tmp_path / "results")]

    def test_echoes_the_output_directory(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path)

        assert f"Wrote experiment results to {tmp_path / 'results'}" in plain(result.output)

    def test_echoes_the_result_repr(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path)

        assert "STUB-EXPERIMENT-REPR" in plain(result.output)
