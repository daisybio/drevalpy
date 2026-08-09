"""Leave-Pair-Out splitting function."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from drevalpy.data.structures import MuDataLike, SplitMask, SplitMasks

from .registry import splitter_registry


@splitter_registry.register("LPO", "Leave-Pair-Out: groups by (cell_line, drug) pairs", validation="LPO")
def leave_pair_out(
    mudataset: MuDataLike,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> list[SplitMasks]:
    """Generate LPO folds where each (cell_line, drug) pair appears in exactly one test set."""
    response = mudataset.response_matrix
    shape = response.shape

    observed = ~np.isnan(response)
    obs_rows, obs_cols = np.where(observed)
    n_observed = len(obs_rows)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: list[SplitMasks] = []

    for train_val_idx, test_idx in kf.split(np.arange(n_observed)):
        n_val = max(1, int(len(train_val_idx) * validation_ratio)) if validation_ratio > 0 else 0
        rng = np.random.default_rng(random_state)
        rng.shuffle(train_val_idx)
        val_idx = train_val_idx[:n_val]
        train_idx = train_val_idx[n_val:]

        train_mask = np.zeros(shape, dtype=bool)
        train_mask[obs_rows[train_idx], obs_cols[train_idx]] = True

        test_mask = np.zeros(shape, dtype=bool)
        test_mask[obs_rows[test_idx], obs_cols[test_idx]] = True

        val_mask = np.zeros(shape, dtype=bool)
        if n_val > 0:
            val_mask[obs_rows[val_idx], obs_cols[val_idx]] = True

        folds.append(SplitMasks(train=SplitMask(train_mask), test=SplitMask(test_mask), val=SplitMask(val_mask)))

    return folds
