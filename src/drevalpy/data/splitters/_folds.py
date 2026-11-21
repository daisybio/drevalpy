"""Shared fold construction for the built-in splitters.

The modes differ only in *what* they hold out - the leakage groups of
:func:`~drevalpy.registry.splitter._groups.pair_groups`: source samples, drugs, tissues,
studies or (source sample, drug) pairs - so the observation mask, the train/validation/test
partition over the groups (k-fold in :func:`grouped_folds`, one test group per fold in
:func:`group_out_folds`), the validation draw and the mask assembly all live here. Keeping the
validation draw in :func:`validation_split` also keeps the folds
reproducible across modes: the validation slice is drawn with a generator seeded from
``random_state`` on every fold, which is a property of this module rather than of any one
splitter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drevalpy.errors import UnknownNameError
from drevalpy.registry.splitter._groups import MODES, in_groups, pair_groups
from drevalpy.types import Fold, PairMask

if TYPE_CHECKING:
    from collections.abc import Iterator

    from drevalpy.data.quality import CurveQuality
    from drevalpy.registry.splitter import Validation
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


def grouped_folds(
    dataset: Dataset,
    quality: CurveQuality,
    *,
    mode: Validation,
    n_splits: int,
    validation_ratio: float,
    random_state: int,
) -> list[Fold]:
    """Build the folds of *mode* by assigning whole leakage groups (:func:`pair_groups`) to the parts.

    The groups that have an observed pair are distributed over *n_splits* test sets with
    :func:`group_folds`.

    :param dataset: Dataset to split.
    :param quality: Curve-quality rule a pair must pass, see :func:`observed_mask`.
    :param mode: Leakage constraint whose groups are held out.
    :param n_splits: Number of folds.
    :param validation_ratio: Fraction of the non-test groups held out for validation.
    :param random_state: Seed of the folds and the validation draw.
    :returns: One ``Fold`` per fold.
    """
    observed = observed_mask(dataset, quality)
    groups = pair_groups(dataset, mode)
    present = np.unique(groups[observed])
    return [
        Fold(
            train=PairMask(observed & in_groups(groups, present[train])),
            test=PairMask(observed & in_groups(groups, present[test])),
            val=PairMask(observed & in_groups(groups, present[validation])),
        )
        for train, validation, test in group_folds(
            len(present), n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state
        )
    ]


def group_out_folds(
    dataset: Dataset, quality: CurveQuality, *, mode: Validation, validation_ratio: float, random_state: int
) -> list[Fold]:
    """Test on each group of *mode* in turn, and draw validation from the other groups by the mode's cut.

    The validation share of the other groups' pairs is drawn by the groups of
    :attr:`~drevalpy.registry.splitter._groups.LeakageGroups.cut` (for LStO: by (source sample,
    drug)), so a sample measured in two of the training studies is never in both training and
    validation.

    :param dataset: Dataset to split.
    :param quality: Curve-quality rule a pair must pass, see :func:`observed_mask`.
    :param mode: Leakage constraint whose groups are tested on, one per fold in group-ID order.
    :param validation_ratio: Fraction of the other groups' cut groups held out for validation.
    :param random_state: Seed of the validation draw.
    :returns: One ``Fold`` per group.
    """
    observed = observed_mask(dataset, quality)
    tested = pair_groups(dataset, mode)
    cut = pair_groups(dataset, MODES[mode].cut)
    folds = []
    for group in range(tested.max() + 1):
        in_group = tested == group
        rest = observed & ~in_group
        _, validation = validation_split(np.unique(cut[rest]), validation_ratio, random_state)
        val = rest & in_groups(cut, validation)
        folds.append(Fold(train=PairMask(rest & ~val), test=PairMask(observed & in_group), val=PairMask(val)))
    return folds
