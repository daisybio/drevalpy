"""Tests for robustness."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.experiment import robustness
from drevalpy.types import SplitMask, SplitMasks


@pytest.fixture
def split_masks() -> SplitMasks:
    rng = np.random.default_rng(0)
    train = rng.random((6, 5)) < 0.6
    test = (~train) & (rng.random((6, 5)) < 0.5)
    return SplitMasks(
        train=SplitMask(train),
        test=SplitMask(test),
        val=SplitMask(~train & ~test),
        metadata={"fold_index": 3, "split_mode": "LPO"},
    )


def test_variant_count_and_metadata(split_masks: SplitMasks) -> None:
    assert robustness(split_masks, 0) == []
    variants = robustness(split_masks, 3)
    assert [v.metadata["robustness_trial"] for v in variants] == [0, 1, 2]
    assert variants[0].metadata["fold_index"] == 3
    assert variants[0].metadata["split_mode"] == "LPO"
    assert "robustness_trial" not in split_masks.metadata


def test_only_pair_order_changes_and_trial_index_seeds_the_shuffle(split_masks: SplitMasks) -> None:
    first, _, third = robustness(split_masks, 3)
    for name in ("train", "test", "val"):
        np.testing.assert_array_equal(getattr(third, name).mask, getattr(split_masks, name).mask)
        np.testing.assert_array_equal(getattr(third, name).pairs, getattr(split_masks, name).shuffled(seed=2).pairs)
    assert not np.array_equal(third.train.pairs, split_masks.train.pairs)
    assert not np.array_equal(first.train.pairs, third.train.pairs)
