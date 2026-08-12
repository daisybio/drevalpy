"""Tests for robustness-trial generation via pair-order shuffling."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.experiment import robustness
from drevalpy.types import SplitMask, SplitMasks


@pytest.fixture
def split_masks() -> SplitMasks:
    rng = np.random.default_rng(0)
    shape = (6, 5)
    train = rng.random(shape) < 0.6
    test = (~train) & (rng.random(shape) < 0.5)
    val = ~train & ~test
    return SplitMasks(
        train=SplitMask(train),
        test=SplitMask(test),
        val=SplitMask(val),
        metadata={"fold_index": 3, "split_mode": "LPO"},
    )


def test_generates_one_variant_per_permutation(split_masks: SplitMasks) -> None:
    assert len(robustness(split_masks, 4)) == 4


def test_zero_permutations_yields_nothing(split_masks: SplitMasks) -> None:
    assert robustness(split_masks, 0) == []


def test_each_variant_records_its_trial_index(split_masks: SplitMasks) -> None:
    variants = robustness(split_masks, 3)

    assert [v.metadata["robustness_trial"] for v in variants] == [0, 1, 2]


def test_original_metadata_is_carried_over(split_masks: SplitMasks) -> None:
    variant = robustness(split_masks, 1)[0]

    assert variant.metadata["fold_index"] == 3
    assert variant.metadata["split_mode"] == "LPO"


def test_original_metadata_is_not_mutated(split_masks: SplitMasks) -> None:
    robustness(split_masks, 2)

    assert "robustness_trial" not in split_masks.metadata


def test_mask_contents_are_unchanged(split_masks: SplitMasks) -> None:
    variant = robustness(split_masks, 1)[0]

    np.testing.assert_array_equal(variant.train.mask, split_masks.train.mask)
    np.testing.assert_array_equal(variant.test.mask, split_masks.test.mask)
    np.testing.assert_array_equal(variant.val.mask, split_masks.val.mask)


def test_pair_order_differs_from_the_original(split_masks: SplitMasks) -> None:
    variant = robustness(split_masks, 2)[1]

    assert not np.array_equal(variant.train.pairs, split_masks.train.pairs)


def test_trial_index_is_used_as_the_shuffle_seed(split_masks: SplitMasks) -> None:
    variant = robustness(split_masks, 3)[2]

    np.testing.assert_array_equal(variant.train.pairs, split_masks.train.shuffled(seed=2).pairs)
    np.testing.assert_array_equal(variant.test.pairs, split_masks.test.shuffled(seed=2).pairs)
    np.testing.assert_array_equal(variant.val.pairs, split_masks.val.shuffled(seed=2).pairs)


def test_distinct_trials_produce_distinct_orders(split_masks: SplitMasks) -> None:
    first, second = robustness(split_masks, 2)

    assert not np.array_equal(first.train.pairs, second.train.pairs)
