"""Robustness testing via pair-order shuffling."""

from __future__ import annotations

from drevalpy.types import Fold


def robustness(fold: Fold, n_repeats: int) -> list[Fold]:
    """Generate shuffled copies of *fold* for robustness testing.

    Each returned Fold has the same mask content but with .pairs in a
    different random order (controlled by the repeat index as seed). This tests
    model stability across different data presentation orders.

    :param fold: Original fold.
    :param n_repeats: Number of robustness repeats.
    :returns: List of Fold with shuffled pair ordering, one per repeat.
    """
    return [
        Fold(
            train=fold.train.shuffled(seed=repeat),
            test=fold.test.shuffled(seed=repeat),
            val=fold.val.shuffled(seed=repeat),
            metadata={**fold.metadata, "robustness_repeat": repeat},
        )
        for repeat in range(n_repeats)
    ]
