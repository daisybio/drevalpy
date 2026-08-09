"""Robustness testing via pair-order shuffling."""

from __future__ import annotations

from drevalpy.types import SplitMasks


def shuffled_splits(split_masks: SplitMasks, n_permutations: int) -> list[SplitMasks]:
    """Generate shuffled copies of split_masks for robustness testing.

    Each returned SplitMasks has the same mask content but with .pairs in a
    different random order (controlled by trial index as seed). This tests
    model stability across different data presentation orders.

    :param split_masks: Original fold split masks.
    :param n_permutations: Number of shuffled variants to generate.
    :returns: List of SplitMasks with shuffled pair ordering, one per trial.
    """
    return [
        SplitMasks(
            train=split_masks.train.shuffled(seed=trial),
            test=split_masks.test.shuffled(seed=trial),
            val=split_masks.val.shuffled(seed=trial),
            metadata={**split_masks.metadata, "robustness_trial": trial},
        )
        for trial in range(n_permutations)
    ]
