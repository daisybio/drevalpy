"""Leave-Pair-Out splitting function."""

from __future__ import annotations

import numpy as np

from drevalpy.data.splitters._folds import group_folds, observed_mask, pair_masks
from drevalpy.registry.splitter import register
from drevalpy.types import MuDataLike, SplitMasks


@register("LPO", "Leave-Pair-Out: groups by (cell_line, drug) pairs", validation="LPO")
def leave_pair_out(
    mudataset: MuDataLike,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> list[SplitMasks]:
    """Generate LPO folds where each (cell_line, drug) pair appears in exactly one test set."""
    observed = observed_mask(mudataset)
    obs_rows, obs_cols = np.where(observed)
    folds = group_folds(
        len(obs_rows),
        n_splits=n_splits,
        validation_ratio=validation_ratio,
        random_state=random_state,
    )
    return [
        pair_masks(observed.shape, obs_rows, obs_cols, train=train, validation=validation, test=test)
        for train, validation, test in folds
    ]
