"""Splitter example: a leave-sample-out variant with a custom fold count.

A splitter is a plain function, not a class. ``validation=`` picks the leakage
constraint the registry enforces after every call, so a splitter that leaks a
sample into both train and test raises ``SplitValidationError`` rather than
silently producing an optimistic score.

It also shows the curve-quality step every built-in splitter performs:
:func:`~drevalpy.plugin.curve_quality_mask` marks the pairs whose fitted
dose-response curve is trustworthy, and blanking the rest keeps them out of
every fold.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from drevalpy.plugin import MuDataLike, SplitMask, SplitMasks, curve_quality_mask, register_splitter


@register_splitter(
    "TOY_LSO",
    "Leave-Sample-Out with a fixed 80/20 train/validation carve-out",
    validation="LSO",
)
def toy_leave_sample_out(
    mudataset: MuDataLike,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> list[SplitMasks]:
    """Split so that every sample is tested exactly once.

    Args:
        mudataset: Dataset whose response matrix is split.
        n_splits: Number of folds.
        validation_ratio: Ignored; this splitter always holds out a fifth of the
            training samples, which is what makes it worth registering
            separately from the built-in ``LSO``.
        random_state: Seed for the fold assignment.

    Returns:
        One :class:`~drevalpy.plugin.SplitMasks` per fold, each carrying
        train/test/validation masks shaped like the response matrix.
    """
    _ = validation_ratio
    # Blank the pairs whose dose-response fit fails the quality thresholds, so
    # the folds only ever contain curves worth training on. The two default
    # thresholds are the ones the built-in splitters use; ``min_r2`` is an extra
    # this splitter opts into.
    response = mudataset.response_matrix.copy()
    response[~curve_quality_mask(mudataset, min_r2=0.5)] = np.nan
    observed = ~np.isnan(response)
    folds: list[SplitMasks] = []
    rng = np.random.default_rng(random_state)

    for train_rows, test_rows in KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
        np.arange(observed.shape[0])
    ):
        shuffled = rng.permutation(train_rows)
        n_validation = max(1, len(shuffled) // 5)
        folds.append(
            SplitMasks(
                train=SplitMask(_rows_mask(observed, shuffled[n_validation:])),
                test=SplitMask(_rows_mask(observed, test_rows)),
                val=SplitMask(_rows_mask(observed, shuffled[:n_validation])),
            )
        )
    return folds


def _rows_mask(observed: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Keep the measured pairs of *rows* and blank everything else."""
    mask = np.zeros_like(observed)
    mask[rows, :] = observed[rows, :]
    return mask
