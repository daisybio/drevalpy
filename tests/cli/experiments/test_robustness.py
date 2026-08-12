"""Tests for :mod:`drevalpy.cli.experiments.robustness`.

:func:`drevalpy.experiment.robustness` is pure ``SplitMasks``
shuffling, so it runs unpatched here and only the empty-directory guard needs
special setup.
"""

from __future__ import annotations

import numpy as np
import pytest
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.types import SplitMask, SplitMasks
from tests.cli._helpers import HELP_ENV, make_runner, plain

runner = make_runner()

N_PERMUTATIONS_DEFAULT = 5


def _write_fold(path: UPath, fold_index: int) -> None:
    """Write a 3x3 fold with enough pairs for shuffling to be observable."""
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
    """A directory holding two fold .npz files."""
    path = tmp_path / "splits"
    path.mkdir()
    for fold_index in range(2):
        _write_fold(path / f"fold_{fold_index}.npz", fold_index)
    return path


def _invoke(splits_dir: UPath, out_dir: UPath, *extra: str):
    return runner.invoke(app, ["experiments", "robustness", str(splits_dir), str(out_dir), *extra])


class TestArguments:
    """Both positional arguments are required."""

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(["experiments", "robustness"], id="none"),
            pytest.param(["experiments", "robustness", "splits"], id="missing-output-dir"),
        ],
    )
    def test_missing_positional_arguments_are_usage_errors(self, argv: list[str]) -> None:
        result = runner.invoke(app, argv, env=HELP_ENV)

        assert result.exit_code == 2


class TestEmptyInput:
    """An input directory with no folds is an explicit exit-1 error."""

    def test_empty_directory_exits_one(self, tmp_path: UPath) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        result = _invoke(empty, tmp_path / "out")

        assert result.exit_code == 1

    def test_empty_directory_reports_the_path(self, tmp_path: UPath) -> None:
        """Typer >= 0.26 merges stderr into ``result.output``."""
        empty = tmp_path / "empty"
        empty.mkdir()

        result = _invoke(empty, tmp_path / "out")

        assert f"No .npz files found in {empty}" in plain(result.output)

    def test_directory_without_npz_files_exits_one(self, tmp_path: UPath) -> None:
        splits = tmp_path / "splits"
        splits.mkdir()
        (splits / "notes.txt").write_text("not a fold")

        result = _invoke(splits, tmp_path / "out")

        assert result.exit_code == 1

    def test_output_directory_is_still_created(self, tmp_path: UPath) -> None:
        """``mkdir`` happens before the guard, so the dir exists even on failure."""
        empty = tmp_path / "empty"
        empty.mkdir()
        out_dir = tmp_path / "out"

        _invoke(empty, out_dir)

        assert out_dir.is_dir()


class TestGeneration:
    """One shuffled variant per (fold, trial) pair."""

    def test_exits_cleanly(self, splits_dir: UPath, tmp_path: UPath) -> None:
        result = _invoke(splits_dir, tmp_path / "out")

        assert result.exit_code == 0, result.output

    def test_creates_the_output_directory_including_parents(self, splits_dir: UPath, tmp_path: UPath) -> None:
        out_dir = tmp_path / "nested" / "out"
        _invoke(splits_dir, out_dir)

        assert out_dir.is_dir()

    def test_default_permutation_count(self, splits_dir: UPath, tmp_path: UPath) -> None:
        out_dir = tmp_path / "out"
        _invoke(splits_dir, out_dir)

        assert len(list(out_dir.glob("*.npz"))) == 2 * N_PERMUTATIONS_DEFAULT

    @pytest.mark.parametrize("flag", ["--n-permutations", "-n"], ids=["long", "short"])
    def test_permutation_count_option(self, splits_dir: UPath, tmp_path: UPath, flag: str) -> None:
        out_dir = tmp_path / "out"
        _invoke(splits_dir, out_dir, flag, "2")

        assert len(list(out_dir.glob("*.npz"))) == 4

    def test_filenames_carry_the_fold_stem_and_trial_index(self, splits_dir: UPath, tmp_path: UPath) -> None:
        out_dir = tmp_path / "out"
        _invoke(splits_dir, out_dir, "-n", "2")

        written = sorted(p.name for p in out_dir.glob("*.npz"))
        assert written == [
            "fold_0_trial_0.npz",
            "fold_0_trial_1.npz",
            "fold_1_trial_0.npz",
            "fold_1_trial_1.npz",
        ]

    def test_variants_record_their_trial_index(self, splits_dir: UPath, tmp_path: UPath) -> None:
        out_dir = tmp_path / "out"
        _invoke(splits_dir, out_dir, "-n", "2")

        variant = SplitMasks.load(out_dir / "fold_0_trial_1.npz")
        assert variant.metadata["robustness_trial"] == 1

    def test_variants_preserve_the_original_metadata(self, splits_dir: UPath, tmp_path: UPath) -> None:
        out_dir = tmp_path / "out"
        _invoke(splits_dir, out_dir, "-n", "1")

        variant = SplitMasks.load(out_dir / "fold_1_trial_0.npz")
        assert variant.metadata["fold_index"] == 1

    def test_variants_preserve_mask_content(self, splits_dir: UPath, tmp_path: UPath) -> None:
        out_dir = tmp_path / "out"
        _invoke(splits_dir, out_dir, "-n", "1")

        original = SplitMasks.load(splits_dir / "fold_0.npz")
        variant = SplitMasks.load(out_dir / "fold_0_trial_0.npz")
        np.testing.assert_array_equal(variant.train.mask, original.train.mask)

    def test_zero_permutations_writes_nothing(self, splits_dir: UPath, tmp_path: UPath) -> None:
        out_dir = tmp_path / "out"
        _invoke(splits_dir, out_dir, "-n", "0")

        assert list(out_dir.glob("*.npz")) == []

    def test_echoes_the_variant_count_and_destination(self, splits_dir: UPath, tmp_path: UPath) -> None:
        out_dir = tmp_path / "out"
        result = _invoke(splits_dir, out_dir, "-n", "3")

        assert f"Wrote 6 robustness splits to {out_dir}" in plain(result.output)

    def test_non_integer_permutation_count_is_a_usage_error(self, splits_dir: UPath, tmp_path: UPath) -> None:
        result = _invoke(splits_dir, tmp_path / "out", "-n", "several")

        assert result.exit_code == 2
