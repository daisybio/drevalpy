"""Tests for :mod:`drevalpy.cli.train`, the ``drevalpy train`` command."""

from __future__ import annotations

import json

import pytest
from sklearn.preprocessing import StandardScaler
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.data.quality import CurveQuality
from tests.cli._helpers import HELP_ENV, FakeDataset, Recorder, make_runner, patch_worker, plain

runner = make_runner()


class StubModel:
    """Stand-in for the fitted ``DRPModel`` that :func:`drevalpy.train` returns."""

    name = "StubModel"

    def __init__(self) -> None:
        """Start with an empty save log."""
        self.saved: list[str] = []

    def save(self, path: UPath) -> None:
        """Record the checkpoint path the command chose.

        :param path: Destination of the checkpoint.
        """
        self.saved.append(str(path))


@pytest.fixture()
def worker(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Patch every lazy import of ``train_cmd``.

    :param monkeypatch: Fixture used to replace the source-module workers.
    :returns: Recorder standing in for :func:`drevalpy.train`.
    """
    recorder = Recorder(return_value=StubModel())
    monkeypatch.setattr("drevalpy.models.ModelSpec.from_name", lambda name: f"spec:{name}")
    monkeypatch.setattr("drevalpy.data.load", lambda name: FakeDataset(name="StubDataset"))
    patch_worker(monkeypatch, "drevalpy._train", "train", recorder)
    return recorder


def _invoke(tmp_path: UPath, *extra: str):
    return runner.invoke(app, ["train", "ElasticNet", "-d", "toy.h5mu", "-o", str(tmp_path / "model"), *extra])


class TestArguments:
    """The model, ``--dataset`` and ``--output`` are required."""

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(["train"], id="none"),
            pytest.param(["train", "ElasticNet"], id="model-only"),
            pytest.param(["train", "ElasticNet", "-d", "toy.h5mu"], id="missing-output"),
            pytest.param(["train", "ElasticNet", "-o", "model"], id="missing-dataset"),
        ],
    )
    def test_missing_arguments_are_usage_errors(self, worker: Recorder, argv: list[str]) -> None:
        result = runner.invoke(app, argv, env=HELP_ENV)

        assert result.exit_code == 2
        assert worker.call_count == 0

    def test_help_lists_the_options(self) -> None:
        output = plain(runner.invoke(app, ["train", "-h"], env=HELP_ENV).output)

        for option in ("--dataset", "--output", "--hpo", "--seed", "--response-transformation", "--no-quality-filter"):
            assert option in output


class TestForwarding:
    """Options map onto :func:`drevalpy.train` keywords."""

    def test_forwards_the_defaults(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path)

        assert result.exit_code == 0, result.output
        spec, dataset = worker.args
        assert spec == "spec:ElasticNet"
        assert dataset.name == "StubDataset"
        forwarded = dict(worker.kwargs)
        assert forwarded.pop("response_transformation") is None
        assert forwarded.pop("quality") == CurveQuality.default()
        assert forwarded == {
            "hyperparameter_tuning": True,
            "hpo_metric": "RMSE",
            "hpo_num_trials": 16,
            "hpo_random_state": None,
            "seed": 42,
            "n_jobs": 1,
        }

    def test_forwards_overridden_options(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(
            tmp_path,
            "--no-hpo",
            "--hpo-metric",
            "Pearson",
            "--hpo-trials",
            "2",
            "--hpo-random-state",
            "9",
            "--seed",
            "3",
        )

        assert worker.kwargs["hyperparameter_tuning"] is False
        assert worker.kwargs["hpo_metric"] == "Pearson"
        assert worker.kwargs["hpo_num_trials"] == 2
        assert worker.kwargs["hpo_random_state"] == 9
        assert worker.kwargs["seed"] == 3

    def test_a_response_transformation_is_forwarded_unfitted(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--response-transformation", "standard")

        transformer = worker.kwargs["response_transformation"]
        assert isinstance(transformer, StandardScaler)
        assert not hasattr(transformer, "mean_")

    def test_the_quality_options_build_the_rule(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--no-quality-filter")
        _invoke(tmp_path, "--min-abs-fold-change", "1.5")

        assert worker.calls[0][1]["quality"] == CurveQuality.off()
        assert worker.calls[1][1]["quality"].min_abs_fold_change == 1.5


class TestOutput:
    """The model directory holds the checkpoint and a record of how it was trained."""

    def test_saves_the_checkpoint_into_the_model_directory(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        assert worker.return_value.saved == [str(tmp_path / "model" / "model.zip")]

    def test_writes_the_training_record(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--no-hpo", "--seed", "5", "--response-transformation", "minmax")

        record = json.loads((tmp_path / "model" / "metadata.json").read_text())
        assert record["model_name"] == "StubModel"
        assert record["dataset_name"] == "StubDataset"
        assert record["seed"] == 5
        assert record["response_transformation"] == "minmax"
        assert record["hyperparameter_tuning"] is False
        assert record["drevalpy_version"]

    def test_echoes_the_destination(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path)

        assert f"StubModel trained on StubDataset -> {tmp_path / 'model'}" in result.output


class TestErrors:
    """Mistakes are reported as one ``Error:`` line by the root handler."""

    def test_an_unknown_model_fails_before_the_dataset_is_read(self, tmp_path: UPath) -> None:
        result = runner.invoke(app, ["train", "NoSuchModel", "-d", "missing.h5mu", "-o", str(tmp_path / "m")])

        assert result.exit_code == 1
        assert "Error: " in result.output
        assert "NoSuchModel" in result.output
        assert "Traceback" not in result.output

    def test_an_unknown_dataset_is_an_error_line(self, tmp_path: UPath) -> None:
        result = runner.invoke(app, ["train", "NaivePredictor", "-d", "no_such_dataset", "-o", str(tmp_path / "m")])

        assert result.exit_code == 1
        assert "Error: Cannot locate .h5mu for dataset 'no_such_dataset'" in result.output
