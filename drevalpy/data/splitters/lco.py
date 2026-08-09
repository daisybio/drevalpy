"""Leave-Cell-Line-Out splitting function."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from drevalpy.data.structures import MuDataLike, SplitMask, SplitMasks

from .registry import splitter_registry


@splitter_registry.register("LCO", "Leave-Cell-Line-Out: test folds contain unseen cell lines", validation="LCO")
def leave_cell_line_out(
    mudataset: MuDataLike,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> list[SplitMasks]:
    """Generate LCO folds where each cell line appears in exactly one test set."""
    response = mudataset.response_matrix
    observed = ~np.isnan(response)
    n_cl = response.shape[0]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: list[SplitMasks] = []

    for train_val_rows, test_rows in kf.split(np.arange(n_cl)):
        n_val = max(1, int(len(train_val_rows) * validation_ratio)) if validation_ratio > 0 else 0
        rng = np.random.default_rng(random_state)
        rng.shuffle(train_val_rows)
        val_rows = train_val_rows[:n_val]
        train_rows = train_val_rows[n_val:]

        train_mask = np.zeros_like(observed)
        train_mask[train_rows, :] = observed[train_rows, :]

        test_mask = np.zeros_like(observed)
        test_mask[test_rows, :] = observed[test_rows, :]

        val_mask = np.zeros_like(observed)
        val_mask[val_rows, :] = observed[val_rows, :]

        folds.append(SplitMasks(train=SplitMask(train_mask), test=SplitMask(test_mask), val=SplitMask(val_mask)))

    return folds
