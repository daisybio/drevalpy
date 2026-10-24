"""Tests for :mod:`drevalpy.cli.experiments.robustness`; the shuffling is pure ``SplitMasks`` work, run unpatched."""

from __future__ import annotations

import numpy as np
import pytest
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.types import SplitMask, SplitMasks
from tests.cli._helpers import HELP_ENV, make_runner, plain, write_synthetic_fold

runner = make_runner()

N_PERMUTATIONS_DEFAULT = 5


def _write_fold(path: UPath, fold_index: int) -> None:
    rng = np.random.default_rng(fold_index)
    assignment = rng.integers(0, 3, size=(3, 3))
    SplitMasks(
        train=SplitMask(assignment == 0),
        test=SplitMask(assignment == 1),
        val=SplitMask(assignment == 2),
        metadata={"fold_index": fold_index},
    ).save(path)


@pytest.fixture()
def splits_dir(tmp_path: UPath) -> UPath:
    path = tmp_path / "splits"
    path.mkdir()
    for fold_index in range(2):
        _write_fold(path / f"fold_{fold_index}.npz", fold_index)
    return path


def _invoke(splits_dir: UPath, out_dir: UPath, *extra: str):
    return runner.invoke(app, ["experiments", "robustness", str(splits_dir), str(out_dir), *extra])


@pytest.mark.parametrize(
    "argv", [["experiments", "robustness"], ["experiments", "robustness", "splits"]], ids=["none", "missing-output-dir"]
)
def test_missing_positional_arguments_are_usage_errors(argv: list[str]) -> None:
    assert runner.invoke(app, argv, env=HELP_ENV).exit_code == 2


@pytest.mark.parametrize("kind", ["empty", "no-npz", "absent"])
def test_a_directory_without_folds_is_an_exit_one_error_and_creates_no_output(tmp_path: UPath, kind: str) -> None:
    splits = tmp_path / "splits"
    if kind != "absent":
        splits.mkdir()
    if kind == "no-npz":
        (splits / "notes.txt").write_text("not a fold")

    result = _invoke(splits, tmp_path / "out")

    assert result.exit_code == 1
    assert f"Error: No .npz files found in {splits}" in plain(result.output)
    assert "Traceback" not in result.output
    assert not (tmp_path / "out").exists()


def test_default_generates_one_variant_per_fold_and_trial(splits_dir: UPath, tmp_path: UPath) -> None:
    out_dir = tmp_path / "nested" / "out"

    result = _invoke(splits_dir, out_dir)

    assert result.exit_code == 0, result.output
    assert len(list(out_dir.glob("*.npz"))) == 2 * N_PERMUTATIONS_DEFAULT


@pytest.mark.parametrize("flag", ["--n-permutations", "-n"])
def test_permutation_count_names_files_after_fold_and_trial_and_echoes_the_count(
    splits_dir: UPath, tmp_path: UPath, flag: str
) -> None:
    out_dir = tmp_path / "out"

    result = _invoke(splits_dir, out_dir, flag, "2")

    assert sorted(p.name for p in out_dir.glob("*.npz")) == [
        "fold_0_trial_0.npz",
        "fold_0_trial_1.npz",
        "fold_1_trial_0.npz",
        "fold_1_trial_1.npz",
    ]
    assert f"Wrote 4 robustness splits to {out_dir}" in plain(result.output)


def test_variants_record_their_trial_and_preserve_metadata_and_masks(splits_dir: UPath, tmp_path: UPath) -> None:
    out_dir = tmp_path / "out"
    _invoke(splits_dir, out_dir, "-n", "2")

    variant = SplitMasks.load(out_dir / "fold_1_trial_1.npz")
    original = SplitMasks.load(splits_dir / "fold_1.npz")
    assert variant.metadata["robustness_trial"] == 1
    assert variant.metadata["fold_index"] == 1
    np.testing.assert_array_equal(variant.train.mask, original.train.mask)


def test_zero_permutations_writes_nothing_and_a_non_integer_count_is_a_usage_error(
    splits_dir: UPath, tmp_path: UPath
) -> None:
    _invoke(splits_dir, tmp_path / "out", "-n", "0")

    assert list((tmp_path / "out").glob("*.npz")) == []
    assert _invoke(splits_dir, tmp_path / "out2", "-n", "several").exit_code == 2


class TestOnRealFiles:
    """A shuffled split trains through ``single`` and is recorded under its robustness trial."""

    pytestmark = pytest.mark.slow

    def test_single_records_the_robustness_trial(self, tmp_path: UPath) -> None:
        from drevalpy.types.results import RunResult

        dataset, fold = write_synthetic_fold(runner, tmp_path)
        shuffled = runner.invoke(app, ["experiments", "robustness", str(fold.parent), str(tmp_path / "rob"), "-n", "2"])
        result = runner.invoke(
            app,
            [
                "single",
                "ElasticNet",
                str(dataset),
                str(tmp_path / "rob" / "fold_0_trial_1.npz"),
                str(tmp_path / "r.npz"),
                "--no-hpo",
            ],
        )

        assert shuffled.exit_code == 0, shuffled.output
        assert result.exit_code == 0, result.output
        run = RunResult.load(str(tmp_path / "r.npz"))
        assert run.robustness_trial == 1
        assert run.randomization is None
