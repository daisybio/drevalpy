"""Built-in cross-validation splitters: LDO, LSO, LPO, LStO, and LTO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drevalpy.data.splitters._folds import group_out_folds, grouped_folds
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
    """Generate LDO folds where each drug appears in exactly one test set."""
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
    """Generate LSO folds where each source sample appears in exactly one test set."""
    return grouped_folds(
        dataset, quality, mode="LSO", n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state
    )


@register("LPO", "Leave-Pair-Out: groups by (sample, drug) pairs", validation="LPO")
def leave_pair_out(
    dataset: Dataset,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    *,
    quality: CurveQuality,
) -> list[Fold]:
    """Generate LPO folds where each (source sample, drug) pair appears in exactly one test set."""
    return grouped_folds(
        dataset, quality, mode="LPO", n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state
    )


@register("LStO", "Leave-Study-Out: test folds contain one unseen study", validation="LStO")
def leave_study_out(
    dataset: Dataset,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    *,
    quality: CurveQuality,
) -> list[Fold]:
    """Generate LStO folds: one fold per study, tested on that study and trained on the others.

    :param dataset: Dataset with a ``study`` sample column (built with ``drevalpy.data.concat``).
    :param n_splits: Ignored; there is always one fold per study.
    :param validation_ratio: Fraction of the training pairs held out for validation.
    :param random_state: Seed for the validation draw.
    :param quality: Curve-quality rule a pair must pass to be used.
    :returns: One ``Fold`` per study.
    :raises KeyError: If the dataset has no ``study`` column.
    :raises ValueError: If the dataset has fewer than two studies.
    """
    try:
        studies = dataset.sample_column("study", dataset.sample_ids).astype(str)
    except KeyError as error:
        raise KeyError(f"LStO needs a 'study' sample column; build it with drevalpy.data.concat. {error}") from error
    unique_studies = np.unique(studies)
    if len(unique_studies) < 2:
        raise ValueError(f"LStO needs at least two studies, found {len(unique_studies)}.")

    folds = group_out_folds(dataset, quality, mode="LStO", validation_ratio=validation_ratio, random_state=random_state)
    for fold, study in zip(folds, unique_studies, strict=True):
        fold.metadata.update(n_splits=len(unique_studies), test_study=str(study))
    return folds


@register("LTO", "Leave-Tissue-Out: test folds contain unseen tissue types", validation="LTO")
def leave_tissue_out(
    dataset: Dataset,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    *,
    quality: CurveQuality,
) -> list[Fold]:
    """Generate LTO folds where each tissue appears in exactly one test set."""
    return grouped_folds(
        dataset, quality, mode="LTO", n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state
    )
