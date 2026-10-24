"""Tests for :mod:`drevalpy.cli.predict`, the ``drevalpy predict`` command, and its round trip with ``train``."""

from __future__ import annotations

import numpy as np
import pytest
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.cli.predict import _checkpoint_path
from drevalpy.errors import InputNotFoundError
from drevalpy.types.results.run import RunResult
from tests.cli._helpers import HELP_ENV, make_runner

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


@pytest.mark.parametrize(
    "argv",
    [["predict"], ["predict", "model"], ["predict", "model", "-d", "toy.h5mu"], ["predict", "model", "-o", "out.npz"]],
)
def test_missing_arguments_are_usage_errors(argv: list[str]) -> None:
    assert runner.invoke(app, argv, env=HELP_ENV).exit_code == 2


def test_train_then_predict_writes_a_run_file_for_every_observed_pair(
    model_dir: UPath, dataset_file: str, tmp_path: UPath
) -> None:
    out = tmp_path / "predictions.npz"
    result = runner.invoke(app, ["predict", str(model_dir), "-d", dataset_file, "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert f"-> {out}" in result.output
    predictions = RunResult.load(out)
    assert len(predictions.predictions) == len(predictions.ground_truth) == 456
    assert np.isfinite(predictions.ground_truth).all()
    assert np.isfinite(predictions.predictions).all()
    assert abs(predictions.predictions.mean() - predictions.ground_truth.mean()) < 0.5
    assert predictions.model_name == "NaiveSampleMeanPredictor"
    assert predictions.dataset_name == "SYNTHETIC"
    assert predictions.metrics


def test_a_missing_npz_suffix_is_added_and_the_checkpoint_file_itself_is_accepted(
    model_dir: UPath, dataset_file: str, tmp_path: UPath
) -> None:
    suffixless = runner.invoke(
        app, ["predict", str(model_dir), "-d", dataset_file, "-o", str(tmp_path / "sub" / "out")]
    )
    from_file = runner.invoke(
        app, ["predict", str(model_dir / "model.zip"), "-d", dataset_file, "-o", str(tmp_path / "out.npz")]
    )

    assert suffixless.exit_code == 0, suffixless.output
    assert (tmp_path / "sub" / "out.npz").is_file()
    assert from_file.exit_code == 0, from_file.output


class TestErrors:
    def test_a_missing_model_directory(self, dataset_file: str, tmp_path: UPath) -> None:
        result = runner.invoke(app, ["predict", str(tmp_path / "nope"), "-d", dataset_file, "-o", str(tmp_path / "o")])

        assert result.exit_code == 1
        assert f"Error: Model directory not found: {tmp_path / 'nope'}" in result.output
        assert "Traceback" not in result.output
        with pytest.raises(InputNotFoundError):
            _checkpoint_path(tmp_path / "nope")

    def test_a_directory_without_a_checkpoint(self, dataset_file: str, tmp_path: UPath) -> None:
        (tmp_path / "empty").mkdir()

        result = runner.invoke(app, ["predict", str(tmp_path / "empty"), "-d", dataset_file, "-o", str(tmp_path / "o")])

        assert result.exit_code == 1
        assert "Error: No model.zip in" in result.output
