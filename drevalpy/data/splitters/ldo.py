"""Leave-Drug-Out splitting function."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split

from drevalpy.data.structures import MuDataLike, SplitMasks

from .registry import splitter_registry


@splitter_registry.register("LDO", "Leave-Drug-Out: test folds contain unseen drugs", validation="LDO")
def leave_drug_out(
    mudataset: MuDataLike,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> list[SplitMasks]:
    """Generate LDO folds where each drug appears in exactly one test set."""
    cl_ids = mudataset.cell_line_ids
    drug_ids = mudataset.drug_ids

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(drug_ids))
    shuffled = drug_ids[perm]

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

        all_cl = np.arange(len(cl_ids))
        train_dr = np.where(np.isin(drug_ids, keep_groups))[0]
        test_dr = np.where(np.isin(drug_ids, test_groups))[0]
        val_dr = np.where(np.isin(drug_ids, val_groups))[0]

        folds.append(
            SplitMasks(
                train_cell_lines=all_cl,
                test_cell_lines=all_cl,
                val_cell_lines=all_cl,
                train_drugs=train_dr,
                test_drugs=test_dr,
                val_drugs=val_dr,
            )
        )

    return folds
