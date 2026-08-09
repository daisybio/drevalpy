"""Leave-Drug-Out splitting function."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from drevalpy.data.structures import MuDataLike, SplitMask, SplitMasks

from .registry import splitter_registry


@splitter_registry.register("LDO", "Leave-Drug-Out: test folds contain unseen drugs", validation="LDO")
def leave_drug_out(
    mudataset: MuDataLike,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> list[SplitMasks]:
    """Generate LDO folds where each drug appears in exactly one test set."""
    response = mudataset.response_matrix
    observed = ~np.isnan(response)
    n_dr = response.shape[1]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: list[SplitMasks] = []

    for train_val_cols, test_cols in kf.split(np.arange(n_dr)):
        n_val = max(1, int(len(train_val_cols) * validation_ratio)) if validation_ratio > 0 else 0
        rng = np.random.default_rng(random_state)
        rng.shuffle(train_val_cols)
        val_cols = train_val_cols[:n_val]
        train_cols = train_val_cols[n_val:]

        train_mask = np.zeros_like(observed)
        train_mask[:, train_cols] = observed[:, train_cols]

        test_mask = np.zeros_like(observed)
        test_mask[:, test_cols] = observed[:, test_cols]

        val_mask = np.zeros_like(observed)
        val_mask[:, val_cols] = observed[:, val_cols]

        folds.append(SplitMasks(train=SplitMask(train_mask), test=SplitMask(test_mask), val=SplitMask(val_mask)))

    return folds
