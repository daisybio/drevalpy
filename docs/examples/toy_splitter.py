"""Splitter example: a leave-cell-line-out variant with a custom fold count.

A splitter is a plain function, not a class. ``validation=`` picks the leakage
constraint the registry enforces after every call, so a splitter that leaks a
cell line into both train and test raises ``SplitValidationError`` rather than
silently producing an optimistic score.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from drevalpy.plugin import MuDataLike, SplitMask, SplitMasks, register_splitter


@register_splitter(
    "TOY_LCO",
    "Leave-Cell-Line-Out with a fixed 80/20 train/validation carve-out",
    validation="LCO",
)
def toy_leave_cell_line_out(
    mudataset: MuDataLike,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> list[SplitMasks]:
    """Split so that every cell line is tested exactly once.

    Args:
        mudataset: Dataset whose response matrix is split.
        n_splits: Number of folds.
        validation_ratio: Ignored; this splitter always holds out a fifth of the
            training cell lines, which is what makes it worth registering
            separately from the built-in ``LCO``.
        random_state: Seed for the fold assignment.

    Returns:
        One :class:`~drevalpy.plugin.SplitMasks` per fold, each carrying
        train/test/validation masks shaped like the response matrix.
    """
    _ = validation_ratio
    observed = ~np.isnan(mudataset.response_matrix)
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
