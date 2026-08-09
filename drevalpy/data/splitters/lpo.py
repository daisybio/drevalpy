"""Leave-Pair-Out splitting function."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split

from drevalpy.data.structures import MuDataLike, SplitMasks

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
    cl_ids = mudataset.cell_line_ids
    drug_ids = mudataset.drug_ids

    # Get a list of all (cell_line, drug) pairs that have a response
    observed_pairs = np.column_stack(np.where(~np.isnan(response)))

    rng = np.random.default_rng(random_state)
    rng.shuffle(observed_pairs)

    # Give all pairs that have the same ids the same label
    pair_labels = np.array([f"{cl_ids[cl]}_{drug_ids[dr]}" for cl, dr in observed_pairs])

    gkf = GroupKFold(n_splits=n_splits)
    folds: list[SplitMasks] = []

    # Split the pairs into n_splits groups, keeping the labels in one group together
    for train_idx, test_idx in gkf.split(observed_pairs, groups=pair_labels):
        if validation_ratio > 0:
            train_idx, val_idx = train_test_split(
                train_idx, test_size=validation_ratio, shuffle=True, random_state=random_state
            )
        else:
            val_idx = np.array([], dtype=np.intp)

        folds.append(
            SplitMasks(
                train=observed_pairs[train_idx],
                test=observed_pairs[test_idx],
                val=observed_pairs[val_idx] if len(val_idx) > 0 else np.empty((0, 2), dtype=np.intp),
            )
        )

    return folds
