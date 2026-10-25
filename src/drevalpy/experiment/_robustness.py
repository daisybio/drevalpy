"""Robustness testing via pair-order shuffling."""

from __future__ import annotations

from drevalpy.types import SplitMasks


def robustness(split_masks: SplitMasks, n_repeats: int) -> list[SplitMasks]:
    """Generate shuffled copies of split_masks for robustness testing.

    Each returned SplitMasks has the same mask content but with .pairs in a
    different random order (controlled by the repeat index as seed). This tests
    model stability across different data presentation orders.

    :param split_masks: Original fold split masks.
    :param n_repeats: Number of robustness repeats.
    :returns: List of SplitMasks with shuffled pair ordering, one per repeat.
    """
    return [
        SplitMasks(
            train=split_masks.train.shuffled(seed=repeat),
            test=split_masks.test.shuffled(seed=repeat),
            val=split_masks.val.shuffled(seed=repeat),
            metadata={**split_masks.metadata, "robustness_repeat": repeat},
        )
        for repeat in range(n_repeats)
    ]
