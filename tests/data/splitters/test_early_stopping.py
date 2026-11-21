"""Tests for :func:`drevalpy.data.splitters._folds.early_stopping_split`."""

from __future__ import annotations

import numpy as np
import pytest

import drevalpy
from drevalpy.data import concat
from drevalpy.data.splitters._folds import early_stopping_split
from drevalpy.errors import InvalidInputError
from drevalpy.experiment import robustness
from drevalpy.registry.splitter._groups import pair_groups
from drevalpy.types import Dataset, Fold, SelectionMask

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return drevalpy.synthetic()


@pytest.fixture(scope="module")
def concatenated() -> Dataset:
    """Two studies of the same 60 samples, so every source sample is on two rows."""
    return concat([drevalpy.synthetic(name="GDSC2"), drevalpy.synthetic(name="CTRPv2", seed=1)])


def _groups_of(groups: np.ndarray, mask: SelectionMask) -> set[int]:
    return set(np.unique(groups[mask.mask]).tolist())


@pytest.mark.parametrize("mode", ["LSO", "LDO", "LTO", "LPO"])
def test_the_stopping_groups_are_in_no_other_part(dataset: Dataset, mode: str) -> None:
    for fold in drevalpy.split(dataset, mode, n_splits=3, validation_ratio=0.5):
        stopping, rest = early_stopping_split(dataset, fold, mode)

        groups = pair_groups(dataset, mode)
        held_out = _groups_of(groups, stopping)
        assert held_out
        for other in (fold.train, rest, fold.test, fold.train | rest):
            assert held_out.isdisjoint(_groups_of(groups, other))
        assert (stopping | rest) == fold.val
        assert not (stopping.mask & rest.mask).any()


def test_a_quarter_of_the_validation_groups_is_held_out(dataset: Dataset) -> None:
    fold = drevalpy.split(dataset, "LSO", n_splits=3, validation_ratio=0.5)[0]
    n_groups = len(_groups_of(pair_groups(dataset, "LSO"), fold.val))

    stopping, _ = early_stopping_split(dataset, fold, "LSO")

    assert len(_groups_of(pair_groups(dataset, "LSO"), stopping)) == max(1, int(n_groups * 0.25))


def test_a_validation_set_of_one_group_raises_and_names_the_unit_the_count_and_the_validation_ratio_options(
    dataset: Dataset,
) -> None:
    fold = drevalpy.split(dataset, "LTO", n_splits=3, validation_ratio=0.1)[0]
    assert len(_groups_of(pair_groups(dataset, "LTO"), fold.val)) == 1

    n_tissues = len(np.unique(pair_groups(dataset, "LTO")))
    with pytest.raises(InvalidInputError) as caught:
        early_stopping_split(dataset, fold, "LTO")

    message = str(caught.value)
    assert "needs at least two tissues in the validation set" in message
    assert f"the validation set of fold 0 has 1 of the dataset's {n_tissues}." in message
    for remedy in (
        "Raise validation_ratio (now 0.1)",
        "drevalpy.run(validation_ratio=...)",
        "drevalpy run --validation-ratio",
        "drevalpy.split(..., validation_ratio=...)",
    ):
        assert remedy in message


def test_the_draw_is_seeded_by_the_fold_and_each_robustness_repeat_draws_other_groups(dataset: Dataset) -> None:
    fold = drevalpy.split(dataset, "LSO", n_splits=3, validation_ratio=0.5)[0]
    reseeded = Fold(fold.train, fold.test, fold.val, {**fold.metadata, "random_state": 1})

    drawn = [early_stopping_split(dataset, variant, "LSO")[0] for variant in robustness(fold, 3)]

    assert early_stopping_split(dataset, fold, "LSO")[0] == early_stopping_split(dataset, fold, "LSO")[0]
    assert early_stopping_split(dataset, reseeded, "LSO")[0] != early_stopping_split(dataset, fold, "LSO")[0]
    assert len({np.flatnonzero(stopping.mask).tobytes() for stopping in drawn}) == 3
    assert [stopping.pair_seed for stopping in drawn] == [0, 1, 2]


def test_lsto_holds_out_source_sample_and_drug_groups(concatenated: Dataset) -> None:
    fold = drevalpy.split(concatenated, "LStO", validation_ratio=0.3)[0]

    stopping, rest = early_stopping_split(concatenated, fold, "LStO")

    pairs = pair_groups(concatenated, "LPO")
    assert _groups_of(pairs, stopping).isdisjoint(_groups_of(pairs, fold.train | rest))
    assert len(_groups_of(pairs, stopping)) == max(1, int(len(_groups_of(pairs, fold.val)) * 0.25))


@pytest.mark.parametrize("mode", ["LSO", "LPO"])
def test_concatenated_studies_keep_a_source_sample_out_of_the_stopping_set_and_the_rest(
    concatenated: Dataset, mode: str
) -> None:
    groups = pair_groups(concatenated, mode)
    for fold in drevalpy.split(concatenated, mode, n_splits=3, validation_ratio=0.5):
        stopping, rest = early_stopping_split(concatenated, fold, mode)

        assert _groups_of(groups, stopping).isdisjoint(_groups_of(groups, fold.train | rest | fold.test))
