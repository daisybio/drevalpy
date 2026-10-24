"""Tests for :mod:`drevalpy.cli.data.split`, the ``drevalpy data split`` command."""

from __future__ import annotations

import numpy as np
import pytest
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.data.quality import CurveQuality
from drevalpy.types import SplitMask, SplitMasks
from drevalpy.types.data.dataset import Dataset
from tests.cli._helpers import HELP_ENV, FakeDataset, Recorder, make_runner, patch_worker, plain

runner = make_runner()

N_FOLDS = 3


def _make_fold(fold_index: int) -> SplitMasks:
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
def loader(monkeypatch: pytest.MonkeyPatch) -> Recorder:

    recorder = Recorder(return_value=FakeDataset())
    patch_worker(monkeypatch, "drevalpy.data.datasets._load", "load", recorder)
    patch_worker(monkeypatch, "drevalpy.data", "load", recorder)
    return recorder


@pytest.fixture()
def splitter(monkeypatch: pytest.MonkeyPatch, loader: Recorder) -> Recorder:

    recorder = Recorder(return_value=[_make_fold(i) for i in range(N_FOLDS)])
    patch_worker(monkeypatch, "drevalpy.data", "split", recorder)
    return recorder


def _invoke(tmp_path: UPath, *extra: str):
    return runner.invoke(app, ["data", "split", "TOYv1", str(tmp_path / "folds"), *extra])


@pytest.mark.parametrize("argv", [["data", "split"], ["data", "split", "TOYv1"]], ids=["none", "missing-output-dir"])
def test_missing_positional_arguments_are_usage_errors(splitter: Recorder, argv: list[str]) -> None:
    assert runner.invoke(app, argv, env=HELP_ENV).exit_code == 2


def test_forwards_the_loaded_dataset_and_defaults(splitter: Recorder, loader: Recorder, tmp_path: UPath) -> None:
    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    assert loader.args == ("TOYv1",)
    assert isinstance(splitter.args[0], FakeDataset)
    assert splitter.kwargs == {
        "mode": "LPO",
        "n_splits": 5,
        "validation_ratio": 0.1,
        "random_state": 42,
        "quality": CurveQuality.default(),
    }


@pytest.mark.parametrize(
    ("flag", "key", "value"),
    [("--mode", "mode", "LSO"), ("-m", "mode", "LSO"), ("--n-splits", "n_splits", 2), ("-n", "n_splits", 2)],
)
def test_short_and_long_options(splitter: Recorder, tmp_path: UPath, flag: str, key: str, value: object) -> None:
    _invoke(tmp_path, flag, str(value))

    assert splitter.kwargs[key] == value


def test_validation_ratio_and_seed_overrides(splitter: Recorder, tmp_path: UPath) -> None:
    _invoke(tmp_path, "--validation-ratio", "0.25", "--seed", "7")

    assert splitter.kwargs["validation_ratio"] == pytest.approx(0.25)
    assert splitter.kwargs["random_state"] == 7


@pytest.mark.parametrize(
    "extra",
    [["--random-state", "7"], ["--validation-ratio", "a-lot"], ["--no-quality-filter", "--min-abs-fold-change", "0.8"]],
    ids=["renamed-seed", "ratio", "threshold-with-no-filter"],
)
def test_invalid_options_are_usage_errors_before_the_splitter(
    splitter: Recorder, tmp_path: UPath, extra: list[str]
) -> None:
    result = _invoke(tmp_path, *extra)

    assert result.exit_code == 2
    assert splitter.call_count == 0


def test_quality_options_build_the_rule(splitter: Recorder, tmp_path: UPath) -> None:
    _invoke(tmp_path, "--no-quality-filter")
    assert splitter.kwargs["quality"] == CurveQuality.off()

    splitter.calls.clear()
    _invoke(tmp_path, "--min-relevance-score", "3", "--min-abs-fold-change", "0.8")
    assert splitter.kwargs["quality"] == CurveQuality(min_relevance_score=3.0, min_abs_fold_change=0.8)


def test_writes_one_reloadable_file_per_fold_and_echoes_the_count(splitter: Recorder, tmp_path: UPath) -> None:
    out_dir = tmp_path / "nested" / "folds"

    result = runner.invoke(app, ["data", "split", "TOYv1", str(out_dir)])

    assert sorted(p.name for p in out_dir.glob("*.npz")) == [f"fold_{i}.npz" for i in range(N_FOLDS)]
    assert SplitMasks.load(out_dir / "fold_1.npz").metadata["fold_index"] == 1
    assert f"Wrote {N_FOLDS} folds to {out_dir}" in plain(result.output)


def test_an_empty_fold_list_still_reports_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: UPath) -> None:
    patch_worker(monkeypatch, "drevalpy.data", "load", Recorder(return_value=FakeDataset()))
    patch_worker(monkeypatch, "drevalpy.data", "split", Recorder(return_value=[]))

    assert f"Wrote 0 folds to {tmp_path / 'folds'}" in plain(_invoke(tmp_path).output)


def test_the_quality_rule_reaches_the_written_folds_and_lco_writes_lso_folds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: UPath, synthetic_dataset: Dataset
) -> None:
    patch_worker(monkeypatch, "drevalpy.data", "load", Recorder(return_value=synthetic_dataset))

    quality = _invoke(tmp_path, "--n-splits", "2", "--min-relevance-score", "4")
    alias = runner.invoke(app, ["data", "split", "TOYv1", str(tmp_path / "lco"), "--mode", "LCO", "-n", "2"])

    assert quality.exit_code == 0, quality.output
    assert alias.exit_code == 0, alias.output
    fold = SplitMasks.load(str(tmp_path / "folds" / "fold_0.npz"))
    assert fold.metadata["quality"] == CurveQuality(min_relevance_score=4.0).to_metadata()
    assert SplitMasks.load(str(tmp_path / "lco" / "fold_0.npz")).metadata["split_mode"] == "LSO"


@pytest.mark.slow
def test_a_cached_dataset_splits_by_name(monkeypatch: pytest.MonkeyPatch, tmp_path: UPath) -> None:
    from drevalpy.data import synthetic

    monkeypatch.setenv("DREVALPY_CACHE_DIR", str(tmp_path / "cache"))
    synthetic().save(tmp_path / "cache" / "CACHED.h5mu")

    result = runner.invoke(app, ["data", "split", "CACHED", str(tmp_path / "folds"), "-n", "2"])

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (tmp_path / "folds").glob("*.npz")) == ["fold_0.npz", "fold_1.npz"]
