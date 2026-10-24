"""Tests for :mod:`drevalpy.cli.single`, the ``drevalpy single`` command."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.types import SplitMask, SplitMasks
from tests.cli._helpers import HELP_ENV, FakeDataset, Recorder, make_runner, patch_worker, plain

runner = make_runner()


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
def worker(monkeypatch: pytest.MonkeyPatch, dataset: FakeDataset) -> Recorder:
    recorder = Recorder(return_value=StubRunResult())
    monkeypatch.setattr("drevalpy.models.ModelSpec.from_name", lambda name: type(f"Stub{name}", (), {}))
    monkeypatch.setattr("drevalpy.types.data.dataset.Dataset.load", classmethod(lambda cls, path: dataset))
    patch_worker(monkeypatch, "drevalpy._single", "single", recorder)
    return recorder


def _invoke(split_file: UPath, tmp_path: UPath, *extra: str):
    out = tmp_path / "out" / "result.npz"
    return runner.invoke(app, ["single", "ElasticNet", str(tmp_path / "ds.h5mu"), str(split_file), str(out), *extra])


@pytest.mark.parametrize(
    "argv", [["single"], ["single", "ElasticNet"], ["single", "ElasticNet", "ds.h5mu", "fold.npz"]]
)
def test_missing_positional_arguments_are_usage_errors(worker: Recorder, argv: list[str]) -> None:
    assert runner.invoke(app, argv, env=HELP_ENV).exit_code == 2


class TestForwarding:
    def test_passes_model_dataset_and_masks_and_forwards_the_defaults(
        self, worker: Recorder, split_file: UPath, tmp_path: UPath, dataset: FakeDataset
    ) -> None:
        result = _invoke(split_file, tmp_path)

        assert result.exit_code == 0, result.output
        model_class, passed_dataset, split_masks = worker.args
        assert model_class.__name__ == "StubElasticNet"
        assert passed_dataset is dataset
        assert isinstance(split_masks, SplitMasks)
        forwarded = dict(worker.kwargs)
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

    def test_forwards_overridden_options(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(
            split_file, tmp_path, "--no-hpo", "--hpo-metric", "Pearson", "--hpo-trials", "2", "--hpo-random-state", "9",
            "--seed", "3", "--n-jobs", "2", "--wandb-project", "my-project", "--split-mode", "LSO",
        )  # fmt: skip

        assert worker.kwargs == {
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
        self, worker: Recorder, split_file: UPath, tmp_path: UPath
    ) -> None:
        _invoke(split_file, tmp_path, "--response-transformation", "standard")
        rejected = _invoke(split_file, tmp_path, "--response-transformation", "logarithmic")

        transformer = worker.kwargs["response_transformation"]
        assert isinstance(transformer, StandardScaler)
        assert not hasattr(transformer, "mean_")
        assert rejected.exit_code == 2
        assert worker.call_count == 1


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
        self, worker: Recorder, tmp_path: UPath, flags: list[str], stored: str | None, expected: str
    ) -> None:
        split_file = tmp_path / "fold.npz"
        _write_split(split_file, metadata={"fold_index": 3, **({"split_mode": stored} if stored else {})})

        _invoke(split_file, tmp_path, *flags)

        assert worker.args[2].metadata["split_mode"] == expected
        assert worker.args[2].metadata["fold_index"] == 3


class TestOutput:
    def test_creates_the_parent_saves_and_echoes_a_summary(
        self, worker: Recorder, split_file: UPath, tmp_path: UPath
    ) -> None:
        result = _invoke(split_file, tmp_path)

        destination = tmp_path / "out" / "result.npz"
        assert (tmp_path / "out").is_dir()
        assert worker.return_value.saved == [str(destination)]
        assert f"Result: StubElasticNet on StubDataset (fold 3) -> {destination}" in plain(result.output)

    def test_a_missing_split_file_is_reported_before_anything_runs(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path / "absent.npz", tmp_path)

        assert result.exit_code == 1
        assert plain(result.output).startswith("Error: ")
        assert "Traceback" not in result.output
        assert worker.call_count == 0
        assert not (tmp_path / "out").exists()
