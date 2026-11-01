"""Split validation: ensures folds satisfy their declared leakage constraints."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from drevalpy.types import Dataset, Fold

Validation = Literal["LSO", "LDO", "LPO", "LTO", "LStO"]


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
    :param dataset: The dataset used for splitting (needed for tissue resolution).
    """
    validator = _VALIDATORS[validation]
    expected_shape = dataset.response_matrix.shape
    for i, fold in enumerate(folds):
        _validate_partition(fold, expected_shape, validation=validation, fold_index=i)
        if validator is not None:
            validator(fold, dataset, fold_index=i)


def _validate_partition(
    fold: Fold,
    expected_shape: tuple[int, ...],
    *,
    validation: Validation,
    fold_index: int,
) -> None:
    """Require three compatible, pairwise-disjoint masks before mode checks."""
    masks = {"train": fold.train, "val": fold.val, "test": fold.test}
    wrong_shapes = {name: mask.shape for name, mask in masks.items() if mask.shape != expected_shape}
    if wrong_shapes:
        raise SplitValidationError(
            f"{validation} validation failed (fold {fold_index}): masks must have response shape "
            f"{expected_shape}; got {wrong_shapes}."
        )

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap_count = int(np.count_nonzero(masks[left].mask & masks[right].mask))
        if overlap_count:
            raise SplitValidationError(
                f"{validation} validation failed (fold {fold_index}): {overlap_count} pairs appear in both "
                f"{left} and {right}."
            )


def _validate_lso(fold: Fold, dataset: Dataset, *, fold_index: int) -> None:
    """LSO: no sample row belongs to more than one split."""
    if _has_axis_overlap(fold, axis=1):
        raise SplitValidationError(
            f"LSO validation failed (fold {fold_index}): sample indices appear in more than one split."
        )


def _validate_ldo(fold: Fold, dataset: Dataset, *, fold_index: int) -> None:
    """LDO: no drug column belongs to more than one split."""
    if _has_axis_overlap(fold, axis=0):
        raise SplitValidationError(
            f"LDO validation failed (fold {fold_index}): drug indices appear in more than one split."
        )


def _validate_lto(fold: Fold, dataset: Dataset, *, fold_index: int) -> None:
    """LTO: no tissue belongs to more than one split."""
    cl_ids = dataset.sample_ids
    tissues = dataset.get_tissue(cl_ids)
    tissue_sets = [
        set(tissues[np.where(mask.mask.any(axis=1))[0]].tolist()) for mask in (fold.train, fold.val, fold.test)
    ]
    overlap = _in_two_splits(*tissue_sets)
    if overlap:
        raise SplitValidationError(
            f"LTO validation failed (fold {fold_index}): "
            f"{len(overlap)} tissues appear in more than one split: {sorted(overlap)[:5]}"
        )


def _validate_lsto(fold: Fold, dataset: Dataset, *, fold_index: int) -> None:
    """LStO: no study has active rows in both test and train/validation."""
    studies = dataset.sample_column("study", dataset.sample_ids)
    train_val, test = (
        set(studies[np.where(mask.mask.any(axis=1))[0]].tolist()) for mask in (fold.train_val, fold.test)
    )
    overlap = train_val & test
    if overlap:
        raise SplitValidationError(
            f"LStO validation failed (fold {fold_index}): "
            f"{len(overlap)} studies appear in both train/validation and test: {sorted(overlap)[:5]}"
        )


def _in_two_splits(train: Any, val: Any, test: Any) -> Any:
    """Return the members (sets) or active entries (boolean arrays) found in at least two splits."""
    return (train & val) | (train & test) | (val & test)


def _has_axis_overlap(fold: Fold, *, axis: int) -> bool:
    """Whether an active row or column occurs in more than one part of the fold."""
    return bool(_in_two_splits(*(mask.mask.any(axis=axis) for mask in (fold.train, fold.val, fold.test))).any())


_VALIDATORS = {
    "LSO": _validate_lso,
    "LDO": _validate_ldo,
    "LTO": _validate_lto,
    "LStO": _validate_lsto,
    # LPO pair disjointness is already enforced by _validate_partition.
    "LPO": None,
}
