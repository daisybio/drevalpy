"""Split validation: ensures folds satisfy their declared leakage constraints.

Every named part of a split (train, validation and test, and the early-stopping set
cut from validation) must have the response shape and be disjoint as pairs. Along the
mode's groups (:func:`~._groups.pair_groups`) the parts must be disjoint too, except
for a mode whose validation is cut by other groups (:attr:`~._groups.LeakageGroups.cut`):
there only the test part must share no group with the others, so under LStO train and
validation may share samples and drugs.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np

from drevalpy.types import Dataset, Fold, SelectionMask

from ._groups import MODES, LeakageGroups, Validation, pair_groups

_N_EXAMPLES = 5


class SplitValidationError(ValueError):
    """Raised when a split violates its declared validation constraints."""


def validate_folds(
    folds: list[Fold],
    validation: Validation,
    dataset: Dataset,
) -> None:
    """Validate all folds against the declared validation constraints.

    Raises ``SplitValidationError`` if any fold violates the constraint.

    :param folds: List of Fold produced by a splitter.
    :param validation: Which leakage constraint to check.
    :param dataset: The dataset used for splitting (needed for the groups of the mode).
    """
    if not folds:
        return
    groups = pair_groups(dataset, validation)
    for i, fold in enumerate(folds):
        parts = {"train": fold.train, "val": fold.val, "test": fold.test}
        _check_parts(parts, validation, dataset, groups, context=f"fold {i}")


def validate_parts(
    parts: Mapping[str, SelectionMask],
    validation: Validation,
    dataset: Dataset,
    *,
    context: str,
) -> None:
    """Validate named parts of one split against the declared validation constraint.

    Raises ``SplitValidationError`` if a part has another shape than the response matrix,
    two parts share a pair, or two parts share a group of the mode (under LStO: the part
    named ``"test"`` shares a study with another part).

    :param parts: Masks by name, such as ``train``, ``stopping``, ``validation`` and ``test``.
    :param validation: Which leakage constraint to check.
    :param dataset: The dataset the masks index.
    :param context: Where the parts come from, for the message (for example ``"fold 2"``).
    """
    _check_parts(parts, validation, dataset, pair_groups(dataset, validation), context=context)


def _check_parts(
    parts: Mapping[str, SelectionMask],
    validation: Validation,
    dataset: Dataset,
    groups: np.ndarray,
    *,
    context: str,
) -> None:
    """Raise ``SplitValidationError`` on the first shape, pair or group violation among *parts*."""
    prefix = f"{validation} validation failed ({context})"
    wrong_shapes = {name: mask.shape for name, mask in parts.items() if mask.shape != groups.shape}
    if wrong_shapes:
        raise SplitValidationError(f"{prefix}: masks must have response shape {groups.shape}; got {wrong_shapes}.")
    for left, right in combinations(parts, 2):
        overlap_count = int(np.count_nonzero(parts[left].mask & parts[right].mask))
        if overlap_count:
            raise SplitValidationError(f"{prefix}: {overlap_count} pairs appear in both {left} and {right}.")
    _check_groups(parts, validation, dataset, groups, prefix=prefix)


def _check_groups(
    parts: Mapping[str, SelectionMask],
    validation: Validation,
    dataset: Dataset,
    groups: np.ndarray,
    *,
    prefix: str,
) -> None:
    """Raise ``SplitValidationError`` if two parts share a group; with a finer cut only pairs with ``test`` count."""
    spec = MODES[validation]
    members = {name: np.unique(groups[mask.mask]) for name, mask in parts.items()}
    checked = [
        (left, right) for left, right in combinations(parts, 2) if spec.cut == validation or "test" in (left, right)
    ]
    for left, right in checked:
        shared = np.intersect1d(members[left], members[right])
        if len(shared):
            examples = sorted(_labels(dataset, spec, groups, shared[:_N_EXAMPLES]))
            raise SplitValidationError(
                f"{prefix}: {len(shared)} {spec.unit} appear in both {left} and {right}: {examples}"
            )


def _labels(dataset: Dataset, spec: LeakageGroups, groups: np.ndarray, shared: np.ndarray) -> list[str]:
    """Name the *shared* groups for a message by the first pair of each."""
    ids, first = np.unique(groups, return_index=True)
    rows, columns = np.unravel_index(first[np.searchsorted(ids, shared)], groups.shape)
    return spec.labels(dataset, rows, columns)
