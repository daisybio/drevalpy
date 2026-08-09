"""Leave-Drug-Out splitting function."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split

from drevalpy.data.structures import MuDataLike, SplitMasks

from .registry import splitter_registry


def _expand_to_pairs(response: np.ndarray, dr_indices: np.ndarray) -> np.ndarray:
    """Expand drug indices to all non-NaN (cl_idx, dr_idx) pairs."""
    if len(dr_indices) == 0:
        return np.empty((0, 2), dtype=np.intp)
    sub = response[:, dr_indices]
    row, col_local = np.where(~np.isnan(sub))
    return np.column_stack([row, dr_indices[col_local]])


@splitter_registry.register("LDO", "Leave-Drug-Out: test folds contain unseen drugs", validation="LDO")
def leave_drug_out(
    mudataset: MuDataLike,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> list[SplitMasks]:
    """Generate LDO folds where each drug appears in exactly one test set."""
    response = mudataset.response_matrix
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

        train_dr = np.where(np.isin(drug_ids, keep_groups))[0]
        test_dr = np.where(np.isin(drug_ids, test_groups))[0]
        val_dr = np.where(np.isin(drug_ids, val_groups))[0]

        folds.append(
            SplitMasks(
                train=_expand_to_pairs(response, train_dr),
                test=_expand_to_pairs(response, test_dr),
                val=_expand_to_pairs(response, val_dr),
            )
        )

    return folds
