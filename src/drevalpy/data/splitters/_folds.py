"""Shared fold construction for the built-in splitters.

The four modes differ only in *what* they hold out - rows, columns, tissues or
individual pairs - so the quality-filtered observation mask, the k-fold
train/validation/test partition and the mask assembly all live here. Keeping the
partition in one place also keeps the folds reproducible across modes: the
validation slice is drawn with a generator seeded from ``random_state`` on every
fold, which is a property of this function rather than of any one splitter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drevalpy.data.quality import curve_quality_mask
from drevalpy.types import SplitMask, SplitMasks

if TYPE_CHECKING:
    from collections.abc import Iterator

    from drevalpy.types import MuDataLike


def observed_mask(mudataset: MuDataLike) -> np.ndarray:
    """Return the pairs that are measured *and* pass the curve-quality filter.

    :param mudataset: Dataset to read the response matrix and quality layers from.
    :returns: Boolean cell-line-by-drug mask of usable pairs.
    """
    response = mudataset.response_matrix.copy()
    response[~curve_quality_mask(mudataset)] = np.nan
    return ~np.isnan(response)


def group_folds(
    n_groups: int,
    *,
    n_splits: int,
    validation_ratio: float,
    random_state: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Partition ``range(n_groups)`` into train, validation and test indices per fold.

    :param n_groups: Number of groups to distribute over the folds.
    :param n_splits: Number of folds; each group is in exactly one test set.
    :param validation_ratio: Fraction of the non-test groups held out for validation.
    :param random_state: Seed for both the fold assignment and the validation draw.
    :returns: One ``(train, validation, test)`` index triple per fold.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_val, test in kf.split(np.arange(n_groups)):
        n_val = max(1, int(len(train_val) * validation_ratio)) if validation_ratio > 0 else 0
        rng = np.random.default_rng(random_state)
        rng.shuffle(train_val)
        yield train_val[n_val:], train_val[:n_val], test


def entity_masks(
    observed: np.ndarray,
    *,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    axis: int,
) -> SplitMasks:
    """Assemble one fold that holds out whole rows (*axis* 0) or columns (*axis* 1).

    :param observed: Mask of usable pairs, as returned by :func:`observed_mask`.
    :param train: Indices along *axis* assigned to training.
    :param validation: Indices along *axis* assigned to validation.
    :param test: Indices along *axis* assigned to testing.
    :param axis: 0 to split cell lines, 1 to split drugs.
    :returns: The three masks of one fold.
    """
    return SplitMasks(
        train=SplitMask(_entity_mask(observed, train, axis)),
        test=SplitMask(_entity_mask(observed, test, axis)),
        val=SplitMask(_entity_mask(observed, validation, axis)),
    )


def pair_masks(
    shape: tuple[int, ...],
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> SplitMasks:
    """Assemble one fold that holds out individual observed pairs.

    :param shape: Shape of the response matrix.
    :param rows: Row coordinate of every observed pair.
    :param columns: Column coordinate of every observed pair, aligned with *rows*.
    :param train: Positions into *rows* / *columns* assigned to training.
    :param validation: Positions assigned to validation.
    :param test: Positions assigned to testing.
    :returns: The three masks of one fold.
    """
    return SplitMasks(
        train=SplitMask(_pair_mask(shape, rows, columns, train)),
        test=SplitMask(_pair_mask(shape, rows, columns, test)),
        val=SplitMask(_pair_mask(shape, rows, columns, validation)),
    )


def rows_with_labels(labels: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Return the row indices whose label is one of *selected*.

    :param labels: One label per row of the response matrix.
    :param selected: Labels belonging to this side of the split.
    :returns: Matching row indices.
    """
    return np.where(np.isin(labels, selected))[0]


def _entity_mask(observed: np.ndarray, indices: np.ndarray, axis: int) -> np.ndarray:
    mask = np.zeros_like(observed)
    if axis == 0:
        mask[indices, :] = observed[indices, :]
    else:
        mask[:, indices] = observed[:, indices]
    return mask


def _pair_mask(
    shape: tuple[int, ...],
    rows: np.ndarray,
    columns: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[rows[positions], columns[positions]] = True
    return mask
