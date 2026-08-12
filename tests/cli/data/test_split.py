"""Tests for :mod:`drevalpy.cli.data.split`, the ``drevalpy data split`` command."""

from __future__ import annotations

import numpy as np
import pytest
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.types import SplitMask, SplitMasks
from tests.cli._helpers import HELP_ENV, FakeDataset, Recorder, make_runner, patch_worker, plain

runner = make_runner()

N_FOLDS = 3


def _make_fold(fold_index: int) -> SplitMasks:
    """Build a real 2x2 fold so the .npz write path is genuine."""
    train = np.zeros((2, 2), dtype=bool)
    train[0, 0] = True
    test = np.zeros((2, 2), dtype=bool)
    test[1, 1] = True
    val = np.zeros((2, 2), dtype=bool)
    val[0, 1] = True
    return SplitMasks(
        train=SplitMask(train),
        test=SplitMask(test),
        val=SplitMask(val),
        metadata={"fold_index": fold_index},
    )


@pytest.fixture()
def splitter(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Patch :func:`drevalpy.data.split` and ``Dataset.load``.

    Args:
        monkeypatch: Fixture used to replace the source-module workers.

    Returns:
        Recorder standing in for the splitter, returning three real folds.
    """
    recorder = Recorder(return_value=[_make_fold(i) for i in range(N_FOLDS)])
    monkeypatch.setattr("drevalpy.types.data.dataset.Dataset.load", classmethod(lambda cls, path: FakeDataset()))
    patch_worker(monkeypatch, "drevalpy.data", "split", recorder)
    return recorder


def _invoke(tmp_path: UPath, *extra: str):
    return runner.invoke(app, ["data", "split", "TOYv1", str(tmp_path / "folds"), *extra])


class TestArguments:
    """Both positional arguments are required."""

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(["data", "split"], id="none"),
            pytest.param(["data", "split", "TOYv1"], id="missing-output-dir"),
        ],
    )
    def test_missing_positional_arguments_are_usage_errors(self, splitter: Recorder, argv: list[str]) -> None:
        result = runner.invoke(app, argv, env=HELP_ENV)

        assert result.exit_code == 2


class TestForwarding:
    """Options map onto :func:`drevalpy.data.split`'s keywords."""

    def test_exits_cleanly(self, splitter: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path)

        assert result.exit_code == 0, result.output

    def test_passes_the_loaded_dataset_positionally(self, splitter: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        assert isinstance(splitter.args[0], FakeDataset)

    def test_defaults(self, splitter: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        assert splitter.kwargs == {
            "mode": "LPO",
            "n_splits": 5,
            "validation_ratio": 0.1,
            "random_state": 42,
        }

    @pytest.mark.parametrize("flag", ["--mode", "-m"], ids=["long", "short"])
    def test_mode(self, splitter: Recorder, tmp_path: UPath, flag: str) -> None:
        _invoke(tmp_path, flag, "LCO")

        assert splitter.kwargs["mode"] == "LCO"

    @pytest.mark.parametrize("flag", ["--n-splits", "-n"], ids=["long", "short"])
    def test_n_splits(self, splitter: Recorder, tmp_path: UPath, flag: str) -> None:
        _invoke(tmp_path, flag, "2")

        assert splitter.kwargs["n_splits"] == 2

    def test_validation_ratio(self, splitter: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--validation-ratio", "0.25")

        assert splitter.kwargs["validation_ratio"] == pytest.approx(0.25)

    def test_random_state(self, splitter: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path, "--random-state", "7")

        assert splitter.kwargs["random_state"] == 7

    def test_non_numeric_validation_ratio_is_a_usage_error(self, splitter: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path, "--validation-ratio", "a-lot")

        assert result.exit_code == 2


class TestOutput:
    """One ``fold_{i}.npz`` per returned fold, plus a summary line."""

    def test_creates_the_output_directory_including_parents(self, splitter: Recorder, tmp_path: UPath) -> None:
        out_dir = tmp_path / "nested" / "folds"
        runner.invoke(app, ["data", "split", "TOYv1", str(out_dir)])

        assert out_dir.is_dir()

    def test_writes_one_file_per_fold(self, splitter: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        written = sorted(p.name for p in (tmp_path / "folds").glob("*.npz"))
        assert written == [f"fold_{i}.npz" for i in range(N_FOLDS)]

    def test_written_folds_are_reloadable(self, splitter: Recorder, tmp_path: UPath) -> None:
        _invoke(tmp_path)

        reloaded = SplitMasks.load(tmp_path / "folds" / "fold_1.npz")
        assert reloaded.metadata["fold_index"] == 1

    def test_echoes_the_fold_count_and_destination(self, splitter: Recorder, tmp_path: UPath) -> None:
        result = _invoke(tmp_path)

        assert f"Wrote {N_FOLDS} folds to {tmp_path / 'folds'}" in plain(result.output)

    def test_empty_fold_list_still_reports_zero(self, monkeypatch: pytest.MonkeyPatch, tmp_path: UPath) -> None:
        monkeypatch.setattr("drevalpy.types.data.dataset.Dataset.load", classmethod(lambda cls, path: FakeDataset()))
        patch_worker(monkeypatch, "drevalpy.data", "split", Recorder(return_value=[]))

        result = _invoke(tmp_path)

        assert f"Wrote 0 folds to {tmp_path / 'folds'}" in plain(result.output)
