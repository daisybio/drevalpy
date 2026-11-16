"""Leave-Drug-Out and Leave-Sample-Out splitting functions, built on :func:`~._folds.grouped_folds`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.data.splitters._folds import grouped_folds
from drevalpy.registry.splitter import register
from drevalpy.types import Dataset, Fold

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
) -> list[Fold]:
    """Generate LDO folds where each drug appears in exactly one test set.

    Every pair with a finite response that passes *quality* is used.
    """
    return grouped_folds(
        dataset, quality, mode="LDO", n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state
    )


@register("LSO", "Leave-Sample-Out: test folds contain unseen samples", validation="LSO", aliases=("LCO",))
def leave_sample_out(
    dataset: Dataset,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    *,
    quality: CurveQuality,
) -> list[Fold]:
    """Generate LSO folds where each source sample appears in exactly one test set.

    Every pair with a finite response that passes *quality* is used. The rows of one source
    sample in a concatenated dataset (``source_sample_id``) are held out together.
    """
    return grouped_folds(
        dataset, quality, mode="LSO", n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state
    )
