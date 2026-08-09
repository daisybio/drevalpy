"""Leave-Pair-Out splitting function."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split

from drevalpy.data.structures import MuDataLike, SplitMasks

from .registry import splitter_registry


@splitter_registry.register("LPO", description="Leave-Pair-Out: groups by (cell_line, drug) pairs")
def leave_pair_out(
    mudataset: MuDataLike,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> list[SplitMasks]:
    """Generate LPO folds where each (cell_line, drug) pair appears in exactly one test set."""
    response = mudataset.response_matrix
    cl_ids = mudataset.cell_line_ids
    drug_ids = mudataset.drug_ids

    row_idx, col_idx = np.where(~np.isnan(response))

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(row_idx))
    row_idx = row_idx[perm]
    col_idx = col_idx[perm]

    groups = np.array([f"{cl_ids[r]}_{drug_ids[c]}" for r, c in zip(row_idx, col_idx, strict=True)])

    gkf = GroupKFold(n_splits=n_splits)
    folds: list[SplitMasks] = []

    for train_pos, test_pos in gkf.split(row_idx, groups=groups):
        if validation_ratio > 0:
            train_pos, val_pos = train_test_split(
                train_pos,
                test_size=validation_ratio,
                shuffle=True,
                random_state=random_state,
            )
        else:
            val_pos = np.array([], dtype=np.intp)

        folds.append(
            SplitMasks(
                train_cell_lines=row_idx[train_pos],
                test_cell_lines=row_idx[test_pos],
                val_cell_lines=row_idx[val_pos],
                train_drugs=col_idx[train_pos],
                test_drugs=col_idx[test_pos],
                val_drugs=col_idx[val_pos],
            )
        )
    return folds
