"""Leave-Study-Out splitting function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drevalpy.data.splitters._folds import group_out_folds
from drevalpy.registry.splitter import register
from drevalpy.types import Dataset, Fold

if TYPE_CHECKING:
    from drevalpy.data.quality import CurveQuality


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

    ``n_splits`` is ignored; the fold metadata records the real number of folds. The
    validation set is a random ``validation_ratio`` share of the other studies' (source sample,
    drug) pairs, so unlike LTO a test sample or drug may also occur in training under another
    study.

    :param dataset: Dataset to read the response matrix and ``study`` sample column from.
    :param n_splits: Ignored; there is always one fold per study.
    :param validation_ratio: Fraction of the training pairs held out for validation.
    :param random_state: Seed for the validation draw.
    :param quality: Curve-quality rule a pair must pass to be used.
    :returns: One ``Fold`` per study.
    :raises KeyError: If the dataset has no ``study`` column; build one with ``drevalpy.data.concat``.
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
