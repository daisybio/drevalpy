"""Leave-Tissue-Out splitting function."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from drevalpy.data.structures import MuDataLike, SplitMask, SplitMasks

from .registry import splitter_registry


@splitter_registry.register("LTO", "Leave-Tissue-Out: test folds contain unseen tissue types", validation="LTO")
def leave_tissue_out(
    mudataset: MuDataLike,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> list[SplitMasks]:
    """Generate LTO folds where each tissue appears in exactly one test set."""
    response = mudataset.response_matrix
    observed = ~np.isnan(response)
    cl_ids = mudataset.cell_line_ids
    tissues = mudataset.get_tissue(cl_ids)

    unique_tissues = np.unique(tissues)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: list[SplitMasks] = []

    for train_val_tissue_idx, test_tissue_idx in kf.split(unique_tissues):
        n_val = max(1, int(len(train_val_tissue_idx) * validation_ratio)) if validation_ratio > 0 else 0
        rng = np.random.default_rng(random_state)
        rng.shuffle(train_val_tissue_idx)
        val_tissue_idx = train_val_tissue_idx[:n_val]
        train_tissue_idx = train_val_tissue_idx[n_val:]

        train_rows = np.where(np.isin(tissues, unique_tissues[train_tissue_idx]))[0]
        test_rows = np.where(np.isin(tissues, unique_tissues[test_tissue_idx]))[0]
        val_rows = np.where(np.isin(tissues, unique_tissues[val_tissue_idx]))[0]

        train_mask = np.zeros_like(observed)
        train_mask[train_rows, :] = observed[train_rows, :]

        test_mask = np.zeros_like(observed)
        test_mask[test_rows, :] = observed[test_rows, :]

        val_mask = np.zeros_like(observed)
        val_mask[val_rows, :] = observed[val_rows, :]

        folds.append(SplitMasks(train=SplitMask(train_mask), test=SplitMask(test_mask), val=SplitMask(val_mask)))

    return folds
