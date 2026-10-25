"""Tests for the ``train``, ``single`` and ``predict`` commands; workers are patched except in the round trip."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.cli.predict import _checkpoint_path
from drevalpy.data.quality import CurveQuality
from drevalpy.errors import InputNotFoundError
from drevalpy.types import SplitMask, SplitMasks
from drevalpy.types.results.run import RunResult
from tests.cli._helpers import HELP_ENV, FakeDataset, Recorder, make_runner, patch_worker, plain

runner = make_runner()


class StubModel:
    name = "StubModel"

    def __init__(self) -> None:
        """Record saved paths."""
        self.saved: list[str] = []

    def save(self, path: UPath) -> None:
        self.saved.append(str(path))


@pytest.fixture()
def train_worker(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder(return_value=StubModel())
    monkeypatch.setattr("drevalpy.models.ModelSpec.from_name", lambda name: f"spec:{name}")
    monkeypatch.setattr("drevalpy.data.load", lambda name: FakeDataset(name="StubDataset"))
    patch_worker(monkeypatch, "drevalpy._train", "train", recorder)
    return recorder


def _invoke_train(tmp_path: UPath, *extra: str):
    return runner.invoke(app, ["train", "ElasticNet", "-d", "toy.h5mu", "-o", str(tmp_path / "model"), *extra])


def test_help_lists_the_options() -> None:
    output = plain(runner.invoke(app, ["train", "-h"], env=HELP_ENV).output)

    for option in ("--dataset", "--output", "--hpo", "--seed", "--response-transformation", "--no-quality-filter"):
        assert option in output


class TestTrainForwarding:
    def test_forwards_the_defaults(self, train_worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke_train(tmp_path)

        assert result.exit_code == 0, result.output
        spec, dataset = train_worker.args
        assert (spec, dataset.name) == ("spec:ElasticNet", "StubDataset")
        forwarded = dict(train_worker.kwargs)
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

    def test_forwards_overridden_options(self, train_worker: Recorder, tmp_path: UPath) -> None:
        _invoke_train(
            tmp_path, "--no-hpo", "--hpo-metric", "Pearson", "--hpo-trials", "2", "--hpo-random-state", "9",
            "--seed", "3", "--response-transformation", "standard", "--min-abs-fold-change", "1.5",
        )  # fmt: skip

        assert train_worker.kwargs["hyperparameter_tuning"] is False
        assert (train_worker.kwargs["hpo_metric"], train_worker.kwargs["hpo_num_trials"]) == ("Pearson", 2)
        assert (train_worker.kwargs["hpo_random_state"], train_worker.kwargs["seed"]) == (9, 3)
        assert isinstance(train_worker.kwargs["response_transformation"], StandardScaler)
        assert not hasattr(train_worker.kwargs["response_transformation"], "mean_")
        assert train_worker.kwargs["quality"].min_abs_fold_change == 1.5

    def test_no_quality_filter_turns_the_rule_off(self, train_worker: Recorder, tmp_path: UPath) -> None:
        _invoke_train(tmp_path, "--no-quality-filter")

        assert train_worker.kwargs["quality"] == CurveQuality.off()


class TestTrainOutput:
    def test_saves_the_checkpoint_the_training_record_and_echoes_the_destination(
        self, train_worker: Recorder, tmp_path: UPath
    ) -> None:
        result = _invoke_train(tmp_path, "--no-hpo", "--seed", "5", "--response-transformation", "minmax")

        assert train_worker.return_value.saved == [str(tmp_path / "model" / "model.zip")]
        record = json.loads((tmp_path / "model" / "metadata.json").read_text())
        assert record["model_name"] == "StubModel"
        assert record["dataset_name"] == "StubDataset"
        assert record["seed"] == 5
        assert record["response_transformation"] == "minmax"
        assert record["hyperparameter_tuning"] is False
        assert record["drevalpy_version"]
        assert f"StubModel trained on StubDataset -> {tmp_path / 'model'}" in result.output


class TestTrainErrors:
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


class StubRunResult:
    model_name = "StubElasticNet"
    dataset_name = "StubDataset"
    fold_index = 3

    def __init__(self) -> None:
        """Record saved paths."""
        self.saved: list[str] = []

    def save(self, path: str) -> None:
        self.saved.append(path)


def _write_split(path: UPath, *, metadata: dict[str, object] | None = None) -> None:
    """Write a real 2x2 ``SplitMasks`` .npz so ``SplitMasks.load`` stays unpatched."""
    train, test, val = (np.zeros((2, 2), dtype=bool) for _ in range(3))
    train[0, 0] = test[1, 1] = val[0, 1] = True
    SplitMasks(
        train=SplitMask(train),
        test=SplitMask(test),
        val=SplitMask(val),
        metadata=dict(metadata or {"fold_index": 3}),
    ).save(path)


@pytest.fixture()
def split_file(tmp_path: UPath) -> UPath:
    path = tmp_path / "fold_0.npz"
    _write_split(path)
    return path


@pytest.fixture()
def dataset() -> FakeDataset:
    return FakeDataset()


@pytest.fixture()
def single_worker(monkeypatch: pytest.MonkeyPatch, dataset: FakeDataset) -> Recorder:
    recorder = Recorder(return_value=StubRunResult())
    monkeypatch.setattr("drevalpy.models.ModelSpec.from_name", lambda name: type(f"Stub{name}", (), {}))
    monkeypatch.setattr("drevalpy.types.data.dataset.Dataset.load", classmethod(lambda cls, path: dataset))
    patch_worker(monkeypatch, "drevalpy._single", "single", recorder)
    return recorder


def _invoke_single(split_file: UPath, tmp_path: UPath, *extra: str):
    out = tmp_path / "out" / "result.npz"
    return runner.invoke(app, ["single", "ElasticNet", str(tmp_path / "ds.h5mu"), str(split_file), str(out), *extra])


class TestSingleForwarding:
    def test_passes_model_dataset_and_masks_and_forwards_the_defaults(
        self, single_worker: Recorder, split_file: UPath, tmp_path: UPath, dataset: FakeDataset
    ) -> None:
        result = _invoke_single(split_file, tmp_path)

        assert result.exit_code == 0, result.output
        model_class, passed_dataset, split_masks = single_worker.args
        assert model_class.__name__ == "StubElasticNet"
        assert passed_dataset is dataset
        assert isinstance(split_masks, SplitMasks)
        forwarded = dict(single_worker.kwargs)
        assert forwarded.pop("response_transformation") is None
        assert forwarded == {
            "hyperparameter_tuning": True,
            "hpo_metric": "RMSE",
            "hpo_num_trials": 16,
            "hpo_random_state": None,
            "seed": 42,
            "wandb_project": None,
            "n_jobs": 1,
        }

    def test_forwards_overridden_options(self, single_worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke_single(
            split_file, tmp_path, "--no-hpo", "--hpo-metric", "Pearson", "--hpo-trials", "2", "--hpo-random-state", "9",
            "--seed", "3", "--n-jobs", "2", "--wandb-project", "my-project", "--split-mode", "LSO",
        )  # fmt: skip

        assert single_worker.kwargs == {
            "hyperparameter_tuning": False,
            "hpo_metric": "Pearson",
            "hpo_num_trials": 2,
            "hpo_random_state": 9,
            "seed": 3,
            "wandb_project": "my-project",
            "n_jobs": 2,
            "response_transformation": None,
        }

    def test_a_response_transformation_is_forwarded_unfitted_and_unknown_ones_rejected(
        self, single_worker: Recorder, split_file: UPath, tmp_path: UPath
    ) -> None:
        _invoke_single(split_file, tmp_path, "--response-transformation", "standard")
        rejected = _invoke_single(split_file, tmp_path, "--response-transformation", "logarithmic")

        transformer = single_worker.kwargs["response_transformation"]
        assert isinstance(transformer, StandardScaler)
        assert not hasattr(transformer, "mean_")
        assert rejected.exit_code == 2
        assert single_worker.call_count == 1


class TestSplitModeInjection:
    """``--split-mode`` is a fallback: it fills the gap, it does not override."""

    @pytest.mark.parametrize(
        ("flags", "stored", "expected"),
        [
            pytest.param(["--split-mode", "LSO"], None, "LSO", id="injected-when-absent"),
            pytest.param(["--split-mode", "LCO"], None, "LSO", id="lco-alias-injected-as-lso"),
            pytest.param([], None, "LPO", id="default-lpo"),
            pytest.param(["--split-mode", "LSO"], "LDO", "LDO", id="existing-metadata-wins"),
            pytest.param([], "LCO", "LSO", id="stored-lco-read-as-lso"),
        ],
    )
    def test_the_split_mode_in_the_metadata(
        self, single_worker: Recorder, tmp_path: UPath, flags: list[str], stored: str | None, expected: str
    ) -> None:
        split_file = tmp_path / "fold.npz"
        _write_split(split_file, metadata={"fold_index": 3, **({"split_mode": stored} if stored else {})})

        _invoke_single(split_file, tmp_path, *flags)

        assert single_worker.args[2].metadata["split_mode"] == expected
        assert single_worker.args[2].metadata["fold_index"] == 3


class TestSingleOutput:
    def test_creates_the_parent_saves_and_echoes_a_summary(
        self, single_worker: Recorder, split_file: UPath, tmp_path: UPath
    ) -> None:
        result = _invoke_single(split_file, tmp_path)

        destination = tmp_path / "out" / "result.npz"
        assert (tmp_path / "out").is_dir()
        assert single_worker.return_value.saved == [str(destination)]
        assert f"Result: StubElasticNet on StubDataset (fold 3) -> {destination}" in plain(result.output)

    def test_a_missing_split_file_is_reported_before_anything_runs(
        self, single_worker: Recorder, tmp_path: UPath
    ) -> None:
        result = _invoke_single(tmp_path / "absent.npz", tmp_path)

        assert result.exit_code == 1
        assert plain(result.output).startswith("Error: ")
        assert "Traceback" not in result.output
        assert single_worker.call_count == 0
        assert not (tmp_path / "out").exists()


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


class TestPredictErrors:
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
