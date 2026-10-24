"""Leave-Sample-Out splitting function."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.data.splitters._folds import entity_masks, group_folds, observed_mask
from drevalpy.registry.splitter import register
from drevalpy.types import MuDataLike, SplitMasks

if TYPE_CHECKING:
    from drevalpy.data.quality import CurveQuality


@register("LSO", "Leave-Sample-Out: test folds contain unseen samples", validation="LSO", aliases=("LCO",))
def leave_sample_out(
    mudataset: MuDataLike,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    *,
    quality: CurveQuality,
) -> list[SplitMasks]:
    """Generate LSO folds where each sample appears in exactly one test set.

    Every pair with a finite response that passes *quality* is used.
    """
    observed = observed_mask(mudataset, quality)
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
