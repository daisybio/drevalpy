"""Leave-Drug-Out splitting function."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.data.splitters._folds import entity_masks, group_folds, observed_mask
from drevalpy.registry.splitter import register
from drevalpy.types import Dataset, SplitMasks

if TYPE_CHECKING:
    from drevalpy.data.quality import CurveQuality


@register("LDO", "Leave-Drug-Out: test folds contain unseen drugs", validation="LDO")
def leave_drug_out(
    dataset: Dataset,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    *,
    quality: CurveQuality,
) -> list[SplitMasks]:
    """Generate LDO folds where each drug appears in exactly one test set.

    Every pair with a finite response that passes *quality* is used.
    """
    observed = observed_mask(dataset, quality)
    folds = group_folds(
        observed.shape[1],
        n_splits=n_splits,
        validation_ratio=validation_ratio,
        random_state=random_state,
    )
    return [
        entity_masks(observed, train=train, validation=validation, test=test, axis=1)
        for train, validation, test in folds
    ]
