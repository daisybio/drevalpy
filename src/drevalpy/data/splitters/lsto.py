"""Leave-Study-Out splitting function."""

from __future__ import annotations

import numpy as np

from drevalpy.data.splitters._folds import observed_mask, pair_masks
from drevalpy.registry.splitter import register
from drevalpy.types import MuDataLike, SplitMasks


@register("LStO", "Leave-Study-Out: test folds contain one unseen study", validation="LStO")
def leave_study_out(
    mudataset: MuDataLike,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> list[SplitMasks]:
    """Generate LStO folds: one fold per study, tested on that study and trained on the others.

    ``n_splits`` is ignored; the fold metadata records the real number of folds. The
    validation set is a random ``validation_ratio`` share of the other studies' pairs, so
    unlike LTO a test sample or drug may also occur in training under another study.

    :raises KeyError: If the dataset has no ``study`` column; build one with ``drevalpy.data.concat``.
    :raises ValueError: If the dataset has fewer than two studies.
    """
    try:
        studies = mudataset.sample_column("study", mudataset.sample_ids).astype(str)
    except KeyError as error:
        raise KeyError(f"LStO needs a 'study' sample column; build it with drevalpy.data.concat. {error}") from error
    unique_studies = np.unique(studies)
    if len(unique_studies) < 2:
        raise ValueError(f"LStO needs at least two studies, found {len(unique_studies)}.")

    observed = observed_mask(mudataset)
    rows, columns = np.where(observed)
    in_study = studies[rows][:, None] == unique_studies[None, :]
    folds = []
    for index, study in enumerate(unique_studies):
        test = np.flatnonzero(in_study[:, index])
        rest = np.flatnonzero(~in_study[:, index])
        n_val = max(1, int(len(rest) * validation_ratio)) if validation_ratio > 0 else 0
        rest = np.random.default_rng(random_state).permutation(rest)
        fold = pair_masks(observed.shape, rows, columns, train=rest[n_val:], validation=rest[:n_val], test=test)
        fold.metadata.update(n_splits=len(unique_studies), test_study=str(study))
        folds.append(fold)
    return folds
