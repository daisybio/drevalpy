"""Leave-Cell-Line-Out splitting function."""

from __future__ import annotations

from drevalpy.data.splitters._folds import entity_masks, group_folds, observed_mask
from drevalpy.registry.splitter import register
from drevalpy.types import MuDataLike, SplitMasks


@register("LCO", "Leave-Cell-Line-Out: test folds contain unseen cell lines", validation="LCO")
def leave_cell_line_out(
    mudataset: MuDataLike,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> list[SplitMasks]:
    """Generate LCO folds where each cell line appears in exactly one test set."""
    observed = observed_mask(mudataset)
    folds = group_folds(
        observed.shape[0],
        n_splits=n_splits,
        validation_ratio=validation_ratio,
        random_state=random_state,
    )
    return [
        entity_masks(observed, train=train, validation=validation, test=test, axis=0)
        for train, validation, test in folds
    ]
