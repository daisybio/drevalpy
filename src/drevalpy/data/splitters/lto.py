"""Leave-Tissue-Out splitting function."""

from __future__ import annotations

import numpy as np

from drevalpy.data.splitters._folds import entity_masks, group_folds, observed_mask, rows_with_labels
from drevalpy.registry.splitter import register
from drevalpy.types import MuDataLike, SplitMasks


@register("LTO", "Leave-Tissue-Out: test folds contain unseen tissue types", validation="LTO")
def leave_tissue_out(
    mudataset: MuDataLike,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> list[SplitMasks]:
    """Generate LTO folds where each tissue appears in exactly one test set."""
    observed = observed_mask(mudataset)
    tissues = mudataset.get_tissue(mudataset.cell_line_ids)
    unique_tissues = np.unique(tissues)
    folds = group_folds(
        len(unique_tissues),
        n_splits=n_splits,
        validation_ratio=validation_ratio,
        random_state=random_state,
    )
    return [
        entity_masks(
            observed,
            train=rows_with_labels(tissues, unique_tissues[train]),
            validation=rows_with_labels(tissues, unique_tissues[validation]),
            test=rows_with_labels(tissues, unique_tissues[test]),
            axis=0,
        )
        for train, validation, test in folds
    ]
