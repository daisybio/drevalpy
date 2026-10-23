"""Tests for :mod:`drevalpy.cli.predict`, the ``drevalpy predict`` command, and its round trip with ``train``."""

from __future__ import annotations

import numpy as np
import pytest
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.cli.predict import _checkpoint_path
from drevalpy.errors import InputNotFoundError
from drevalpy.types.results.run import RunResult
from tests.cli._helpers import HELP_ENV, make_runner, plain

runner = make_runner()


@pytest.fixture(scope="module")
def model_dir(dataset_file: str, tmp_path_factory: pytest.TempPathFactory) -> UPath:
    """Train a response-scaled model through the command line once."""
    directory = UPath(tmp_path_factory.mktemp("cli_model")) / "model"
    result = runner.invoke(
        app,
        ["train", "NaiveSampleMeanPredictor", "-d", dataset_file, "-o", str(directory), "--no-hpo"]
        + ["--response-transformation", "standard"],
    )
    assert result.exit_code == 0, result.output
    return directory


class TestArguments:
    """The model directory, ``--dataset`` and ``--output`` are required."""

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(["predict"], id="none"),
            pytest.param(["predict", "model"], id="model-only"),
            pytest.param(["predict", "model", "-d", "toy.h5mu"], id="missing-output"),
            pytest.param(["predict", "model", "-o", "out.npz"], id="missing-dataset"),
        ],
    )
    def test_missing_arguments_are_usage_errors(self, argv: list[str]) -> None:
        assert runner.invoke(app, argv, env=HELP_ENV).exit_code == 2

    def test_help_lists_the_options(self) -> None:
        output = plain(runner.invoke(app, ["predict", "-h"], env=HELP_ENV).output)

        assert "--dataset" in output
        assert "--output" in output


class TestRoundTrip:
    """``train`` followed by ``predict`` writes a run-layout file for every observed pair."""

    @pytest.fixture(scope="class")
    def predictions(self, model_dir: UPath, dataset_file: str, tmp_path_factory: pytest.TempPathFactory) -> RunResult:
        out = UPath(tmp_path_factory.mktemp("cli_predictions")) / "predictions.npz"
        result = runner.invoke(app, ["predict", str(model_dir), "-d", dataset_file, "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert f"-> {out}" in result.output
        return RunResult.load(out)

    def test_predicts_every_observed_pair(self, predictions: RunResult) -> None:
        assert len(predictions.predictions) == len(predictions.ground_truth) == 456
        assert np.isfinite(predictions.ground_truth).all()

    def test_predictions_are_in_response_units(self, predictions: RunResult) -> None:
        assert np.isfinite(predictions.predictions).all()
        assert abs(predictions.predictions.mean() - predictions.ground_truth.mean()) < 0.5

    def test_names_the_model_and_the_dataset(self, predictions: RunResult) -> None:
        assert predictions.model_name == "NaiveSampleMeanPredictor"
        assert predictions.dataset_name == "SYNTHETIC"
        assert predictions.metrics

    def test_a_missing_npz_suffix_is_added(self, model_dir: UPath, dataset_file: str, tmp_path: UPath) -> None:
        result = runner.invoke(
            app, ["predict", str(model_dir), "-d", dataset_file, "-o", str(tmp_path / "sub" / "out")]
        )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "sub" / "out.npz").is_file()

    def test_the_checkpoint_file_itself_is_accepted(self, model_dir: UPath, dataset_file: str, tmp_path: UPath) -> None:
        result = runner.invoke(
            app, ["predict", str(model_dir / "model.zip"), "-d", dataset_file, "-o", str(tmp_path / "out.npz")]
        )

        assert result.exit_code == 0, result.output


class TestErrors:
    """A model directory that cannot be used is an ``Error:`` line, before the dataset is read."""

    def test_a_missing_model_directory(self, dataset_file: str, tmp_path: UPath) -> None:
        result = runner.invoke(app, ["predict", str(tmp_path / "nope"), "-d", dataset_file, "-o", str(tmp_path / "o")])

        assert result.exit_code == 1
        assert f"Error: Model directory not found: {tmp_path / 'nope'}" in result.output
        assert "Traceback" not in result.output

    def test_a_directory_without_a_checkpoint(self, dataset_file: str, tmp_path: UPath) -> None:
        (tmp_path / "empty").mkdir()

        result = runner.invoke(app, ["predict", str(tmp_path / "empty"), "-d", dataset_file, "-o", str(tmp_path / "o")])

        assert result.exit_code == 1
        assert "Error: No model.zip in" in result.output

    def test_checkpoint_path_raises_the_input_error(self, tmp_path: UPath) -> None:
        with pytest.raises(InputNotFoundError):
            _checkpoint_path(tmp_path / "nope")
