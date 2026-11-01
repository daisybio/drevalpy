"""Leave-Drug-Out and Leave-Sample-Out splitting functions, built on :func:`~._folds.entity_folds`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.data.splitters._folds import entity_folds
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
    return entity_folds(
        dataset, quality, axis=1, n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state
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
    """Generate LSO folds where each sample appears in exactly one test set.

    Every pair with a finite response that passes *quality* is used.
    """
    return entity_folds(
        dataset, quality, axis=0, n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state
    )
