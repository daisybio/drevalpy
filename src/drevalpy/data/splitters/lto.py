"""Leave-Tissue-Out splitting function."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.data.splitters._folds import grouped_folds
from drevalpy.registry.splitter import register
from drevalpy.types import Dataset, Fold

if TYPE_CHECKING:
    from drevalpy.data.quality import CurveQuality


@register("LTO", "Leave-Tissue-Out: test folds contain unseen tissue types", validation="LTO")
def leave_tissue_out(
    dataset: Dataset,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    *,
    quality: CurveQuality,
) -> list[Fold]:
    """Generate LTO folds where each tissue appears in exactly one test set.

    Every pair with a finite response that passes *quality* is used.
    """
    return grouped_folds(
        dataset, quality, mode="LTO", n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state
    )
