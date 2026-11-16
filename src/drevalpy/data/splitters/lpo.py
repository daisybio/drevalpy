"""Leave-Pair-Out splitting function."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.data.splitters._folds import grouped_folds
from drevalpy.registry.splitter import register
from drevalpy.types import Dataset, Fold

if TYPE_CHECKING:
    from drevalpy.data.quality import CurveQuality


@register("LPO", "Leave-Pair-Out: groups by (sample, drug) pairs", validation="LPO")
def leave_pair_out(
    dataset: Dataset,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    *,
    quality: CurveQuality,
) -> list[Fold]:
    """Generate LPO folds where each (source sample, drug) pair appears in exactly one test set.

    Every pair with a finite response that passes *quality* is used. In a concatenated dataset
    the measurements of one sample and drug in several studies (``source_sample_id``) are held
    out together.
    """
    return grouped_folds(
        dataset, quality, mode="LPO", n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state
    )
