"""Tests for :mod:`drevalpy.cli.single`, the ``drevalpy single`` command."""

from __future__ import annotations

import numpy as np
import pytest
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.types import SplitMask, SplitMasks
from tests.cli._helpers import HELP_ENV, FakeDataset, Recorder, make_runner, patch_worker, plain

runner = make_runner()


class StubRunResult:
    """Stand-in for the ``RunResult`` that :func:`drevalpy.single` returns."""

    model_name = "StubElasticNet"
    dataset_name = "StubDataset"
    fold_index = 3

    def __init__(self) -> None:
        """Start with an empty save log."""
        self.saved: list[str] = []

    def save(self, path: str) -> None:
        """Record the destination the command derived.

        Args:
            path: Output file path chosen by the command.
        """
        self.saved.append(path)


def _write_split(path: UPath, *, metadata: dict[str, object] | None = None) -> None:
    """Write a real 2x2 ``SplitMasks`` .npz so ``SplitMasks.load`` stays unpatched."""
    train = np.zeros((2, 2), dtype=bool)
    train[0, 0] = True
    test = np.zeros((2, 2), dtype=bool)
    test[1, 1] = True
    val = np.zeros((2, 2), dtype=bool)
    val[0, 1] = True
    SplitMasks(
        train=SplitMask(train),
        test=SplitMask(test),
        val=SplitMask(val),
        metadata=dict(metadata or {"fold_index": 3}),
    ).save(path)


@pytest.fixture()
def split_file(tmp_path: UPath) -> UPath:
    """A fold .npz whose metadata deliberately lacks ``split_mode``."""
    path = tmp_path / "fold_0.npz"
    _write_split(path)
    return path


@pytest.fixture()
def dataset() -> FakeDataset:
    """The dataset stub ``Dataset.load`` is made to return."""
    return FakeDataset()


@pytest.fixture()
def worker(monkeypatch: pytest.MonkeyPatch, dataset: FakeDataset) -> Recorder:
    """Patch every lazy import of ``single_cmd`` except ``SplitMasks``.

    ``SplitMasks.load`` is left real so the ``split_mode`` injection is asserted
    against the genuine round-trip rather than a stub's attribute.

    Args:
        monkeypatch: Fixture used to replace the source-module workers.
        dataset: Stub returned by the patched ``Dataset.load``.

    Returns:
        Recorder standing in for :func:`drevalpy.single`.
    """
    recorder = Recorder(return_value=StubRunResult())
    monkeypatch.setattr("drevalpy.models.construct_model", lambda name: type(f"Stub{name}", (), {}))
    monkeypatch.setattr("drevalpy.types.data.dataset.Dataset.load", classmethod(lambda cls, path: dataset))
    patch_worker(monkeypatch, "drevalpy._single", "single", recorder)
    return recorder


def _invoke(split_file: UPath, tmp_path: UPath, *extra: str):
    out = tmp_path / "out" / "result.npz"
    return runner.invoke(
        app,
        ["single", "ElasticNet", str(tmp_path / "ds.h5mu"), str(split_file), str(out), *extra],
    )


class TestArguments:
    """All four positional arguments are required."""

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(["single"], id="none"),
            pytest.param(["single", "ElasticNet"], id="model-only"),
            pytest.param(["single", "ElasticNet", "ds.h5mu"], id="missing-split-and-output"),
            pytest.param(["single", "ElasticNet", "ds.h5mu", "fold.npz"], id="missing-output"),
        ],
    )
    def test_missing_positional_arguments_are_usage_errors(self, worker: Recorder, argv: list[str]) -> None:
        result = runner.invoke(app, argv, env=HELP_ENV)

        assert result.exit_code == 2


class TestForwarding:
    """Options map onto :func:`drevalpy.single` keywords."""

    def test_exits_cleanly(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        result = _invoke(split_file, tmp_path)

        assert result.exit_code == 0, result.output

    def test_passes_model_class_dataset_and_masks_positionally(
        self, worker: Recorder, split_file: UPath, tmp_path: UPath, dataset: FakeDataset
    ) -> None:
        _invoke(split_file, tmp_path)

        model_class, passed_dataset, split_masks = worker.args
        assert model_class.__name__ == "StubElasticNet"
        assert passed_dataset is dataset
        assert isinstance(split_masks, SplitMasks)

    def test_forwards_hpo_defaults(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path)

        assert worker.kwargs == {
            "hyperparameter_tuning": True,
            "hpo_metric": "RMSE",
            "hpo_num_samples": 16,
            "hpo_random_state": 42,
        }

    def test_no_hpo_disables_tuning(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path, "--no-hpo")

        assert worker.kwargs["hyperparameter_tuning"] is False

    def test_forwards_overridden_hpo_options(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path, "--hpo-metric", "Pearson", "--hpo-num-samples", "2", "--hpo-random-state", "9")

        assert worker.kwargs["hpo_metric"] == "Pearson"
        assert worker.kwargs["hpo_num_samples"] == 2
        assert worker.kwargs["hpo_random_state"] == 9

    def test_split_mode_is_not_a_worker_keyword(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        """``--split-mode`` only ever reaches the worker through the masks' metadata."""
        _invoke(split_file, tmp_path, "--split-mode", "LCO")

        assert "split_mode" not in worker.kwargs


class TestSplitModeInjection:
    """``--split-mode`` is a fallback: it fills the gap, it does not override."""

    def test_injected_when_absent_from_metadata(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path, "--split-mode", "LCO")

        assert worker.args[2].metadata["split_mode"] == "LCO"

    def test_default_is_lpo(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path)

        assert worker.args[2].metadata["split_mode"] == "LPO"

    def test_existing_metadata_wins(self, worker: Recorder, tmp_path: UPath) -> None:
        split_file = tmp_path / "fold_with_mode.npz"
        _write_split(split_file, metadata={"fold_index": 3, "split_mode": "LDO"})

        _invoke(split_file, tmp_path, "--split-mode", "LCO")

        assert worker.args[2].metadata["split_mode"] == "LDO"

    def test_other_metadata_is_preserved(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path)

        assert worker.args[2].metadata["fold_index"] == 3


class TestOutput:
    """The command creates the output *parent* and echoes a one-line summary."""

    def test_creates_the_parent_directory(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path)

        assert (tmp_path / "out").is_dir()

    def test_saves_to_the_requested_path(self, worker: Recorder, split_file: UPath, tmp_path: UPath) -> None:
        _invoke(split_file, tmp_path)

        assert worker.return_value.saved == [str(tmp_path / "out" / "result.npz")]

    def test_echoes_model_dataset_fold_and_destination(
        self, worker: Recorder, split_file: UPath, tmp_path: UPath
    ) -> None:
        result = _invoke(split_file, tmp_path)

        expected = f"Result: StubElasticNet on StubDataset (fold 3) -> {tmp_path / 'out' / 'result.npz'}"
        assert expected in plain(result.output)


class TestMissingSplitFile:
    """A missing fold file surfaces as the loader's own error, not a silent pass."""

    def test_nonexistent_split_raises(self, worker: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path / "absent.npz", tmp_path)

        assert result.exit_code != 0

    def test_worker_is_not_called(self, worker: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path / "absent.npz", tmp_path)

        assert worker.call_count == 0
