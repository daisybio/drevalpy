"""Shared fold construction for the built-in splitters.

The modes differ only in *what* they hold out - rows, columns, tissues, studies or
individual pairs - so the observation mask, the k-fold train/validation/test
partition, the validation draw and the mask assembly all live here. Keeping the
validation draw in :func:`validation_split` also keeps the folds reproducible
across modes: the validation slice is drawn with a generator seeded from
``random_state`` on every fold, which is a property of this module rather than of
any one splitter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drevalpy.errors import UnknownNameError
from drevalpy.types import Fold, PairMask

if TYPE_CHECKING:
    from collections.abc import Iterator

    from drevalpy.data.quality import CurveQuality
    from drevalpy.types import Dataset


def observed_mask(dataset: Dataset, quality: CurveQuality) -> np.ndarray:
    """Return the pairs with a finite response that also pass the curve-quality rule.

    A splitter calls this with the ``quality`` it was given, so the user's choice
    (``quality=`` in :func:`drevalpy.data.split`, ``--no-quality-filter`` on the command
    line) reaches it. With :meth:`~drevalpy.data.quality.CurveQuality.off` the result is
    plain finiteness and no response layer is read, so datasets without quality layers work.

    :param dataset: Dataset to read the response matrix and, for a rule, the quality layers from.
    :param quality: Curve-quality rule a pair must pass.
    :returns: Boolean sample-by-drug mask of usable pairs.
    :raises UnknownNameError: If *quality* needs a response layer *dataset* lacks (a ``KeyError``).
    """
    try:
        passed = quality.mask(dataset)
    except KeyError as error:
        raise UnknownNameError(
            f"{error.args[0]} The curve-quality rule needs it; pass CurveQuality.off() to skip the filter."
        ) from error
    return np.isfinite(dataset.response_matrix) & passed


def validation_split(indices: np.ndarray, validation_ratio: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle *indices* with a generator seeded from *random_state* and hold out the validation share.

    This is the one validation rule of the built-in splitters: ``max(1, int(n * validation_ratio))``
    indices, or none for a ratio of 0.

    :param indices: The non-test indices of one fold.
    :param validation_ratio: Fraction of *indices* held out for validation.
    :param random_state: Seed of the shuffle.
    :returns: ``(train, validation)`` indices.
    """
    n_val = max(1, int(len(indices) * validation_ratio)) if validation_ratio > 0 else 0
    shuffled = np.random.default_rng(random_state).permutation(indices)
    return shuffled[n_val:], shuffled[:n_val]


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
    :yields: One ``(train, validation, test)`` index triple per fold.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_val, test in kf.split(np.arange(n_groups)):
        train, validation = validation_split(train_val, validation_ratio, random_state)
        yield train, validation, test


def entity_folds(
    dataset: Dataset,
    quality: CurveQuality,
    *,
    axis: int,
    n_splits: int,
    validation_ratio: float,
    random_state: int,
) -> list[Fold]:
    """Build the folds of LSO (*axis* 0, samples) or LDO (*axis* 1, drugs): each entity is in one test set.

    :param dataset: Dataset to split.
    :param quality: Curve-quality rule a pair must pass, see :func:`observed_mask`.
    :param axis: 0 to hold out samples, 1 to hold out drugs.
    :param n_splits: Number of folds.
    :param validation_ratio: Fraction of the non-test entities held out for validation.
    :param random_state: Seed of the folds and the validation draw.
    :returns: One ``Fold`` per fold.
    """
    observed = observed_mask(dataset, quality)
    folds = group_folds(
        observed.shape[axis], n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state
    )
    return [
        entity_masks(observed, train=train, validation=validation, test=test, axis=axis)
        for train, validation, test in folds
    ]


def entity_masks(
    observed: np.ndarray,
    *,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    axis: int,
) -> Fold:
    """Assemble one fold that holds out whole rows (*axis* 0) or columns (*axis* 1).

    :param observed: Mask of usable pairs, as returned by :func:`observed_mask`.
    :param train: Indices along *axis* assigned to training.
    :param validation: Indices along *axis* assigned to validation.
    :param test: Indices along *axis* assigned to testing.
    :param axis: 0 to split samples, 1 to split drugs.
    :returns: The three masks of one fold.
    """

    def mask(indices: np.ndarray) -> PairMask:
        selected = np.zeros_like(observed)
        if axis == 0:
            selected[indices, :] = observed[indices, :]
        else:
            selected[:, indices] = observed[:, indices]
        return PairMask(selected)

    return Fold(train=mask(train), test=mask(test), val=mask(validation))


def pair_masks(
    shape: tuple[int, ...],
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> Fold:
    """Assemble one fold that holds out individual observed pairs.

    :param shape: Shape of the response matrix.
    :param rows: Row coordinate of every observed pair.
    :param columns: Column coordinate of every observed pair, aligned with *rows*.
    :param train: Positions into *rows* / *columns* assigned to training.
    :param validation: Positions assigned to validation.
    :param test: Positions assigned to testing.
    :returns: The three masks of one fold.
    """

    def mask(positions: np.ndarray) -> PairMask:
        selected = np.zeros(shape, dtype=bool)
        selected[rows[positions], columns[positions]] = True
        return PairMask(selected)

    return Fold(train=mask(train), test=mask(test), val=mask(validation))


def rows_with_labels(labels: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Return the row indices whose label is one of *selected*.

    :param labels: One label per row of the response matrix.
    :param selected: Labels belonging to this side of the split.
    :returns: Matching row indices.
    """
    return np.where(np.isin(labels, selected))[0]
