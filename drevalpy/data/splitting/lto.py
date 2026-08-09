"""Leave-Tissue-Out splitting function."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split

from drevalpy.data.structures import MuDataLike, SplitMasks

from .registry import splitter_registry


@splitter_registry.register("LTO", description="Leave-Tissue-Out: test folds contain unseen tissue types")
def leave_tissue_out(
    mudataset: MuDataLike,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> list[SplitMasks]:
    """Generate LTO folds where each tissue appears in exactly one test set."""
    cl_ids = mudataset.cell_line_ids
    tissues = mudataset.get_tissue(cl_ids)

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(tissues))
    shuffled = tissues[perm]

    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(shuffled))
    folds: list[SplitMasks] = []

    for train_pos, test_pos in gkf.split(dummy, groups=shuffled):
        train_groups = shuffled[train_pos]
        test_groups = np.unique(shuffled[test_pos])

        unique_train = np.unique(train_groups)
        if validation_ratio > 0 and len(unique_train) > 1:
            keep_groups, val_groups = train_test_split(
                unique_train,
                test_size=validation_ratio,
                shuffle=True,
                random_state=random_state,
            )
        else:
            keep_groups = unique_train
            val_groups = np.array([], dtype=unique_train.dtype)

        train_cl = np.where(np.isin(tissues, keep_groups))[0]
        test_cl = np.where(np.isin(tissues, test_groups))[0]
        val_cl = np.where(np.isin(tissues, val_groups))[0]

        folds.append(SplitMasks(train_cell_lines=train_cl, test_cell_lines=test_cl, val_cell_lines=val_cl))

    return folds
