"""Split validation: ensures folds satisfy their declared leakage constraints."""

from __future__ import annotations

from typing import Literal

import numpy as np

from drevalpy.types import MuDataLike, SplitMasks

Validation = Literal["LCO", "LDO", "LPO", "LTO"]


class SplitValidationError(ValueError):
    """Raised when a split violates its declared validation constraints."""


def validate_folds(
    folds: list[SplitMasks],
    validation: Validation,
    mudataset: MuDataLike,
) -> None:
    """Validate all folds against the declared validation constraints.

    :param folds: List of SplitMasks produced by a splitter.
    :param validation: Which leakage constraint to check.
    :param mudataset: The dataset used for splitting (needed for tissue resolution).
    :raises SplitValidationError: If any fold violates the constraint.
    """
    validator = _VALIDATORS[validation]
    expected_shape = mudataset.response_matrix.shape
    for i, fold in enumerate(folds):
        _validate_partition(fold, expected_shape, validation=validation, fold_index=i)
        validator(fold, mudataset, fold_index=i)


def _validate_partition(
    fold: SplitMasks,
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


def _validate_lco(fold: SplitMasks, mudataset: MuDataLike, *, fold_index: int) -> None:
    """LCO: no sample row belongs to more than one split."""
    if _has_axis_overlap(fold, axis=1):
        raise SplitValidationError(
            f"LCO validation failed (fold {fold_index}): sample indices appear in more than one split."
        )


def _validate_ldo(fold: SplitMasks, mudataset: MuDataLike, *, fold_index: int) -> None:
    """LDO: no drug column belongs to more than one split."""
    if _has_axis_overlap(fold, axis=0):
        raise SplitValidationError(
            f"LDO validation failed (fold {fold_index}): drug indices appear in more than one split."
        )


def _validate_lto(fold: SplitMasks, mudataset: MuDataLike, *, fold_index: int) -> None:
    """LTO: no tissue belongs to more than one split."""
    cl_ids = mudataset.sample_ids
    tissues = mudataset.get_tissue(cl_ids)
    tissue_sets = [
        set(tissues[np.where(mask.mask.any(axis=1))[0]].tolist()) for mask in (fold.train, fold.val, fold.test)
    ]
    overlap = (tissue_sets[0] & tissue_sets[1]) | (tissue_sets[0] & tissue_sets[2]) | (tissue_sets[1] & tissue_sets[2])
    if overlap:
        raise SplitValidationError(
            f"LTO validation failed (fold {fold_index}): "
            f"{len(overlap)} tissues appear in more than one split: {sorted(overlap)[:5]}"
        )


def _validate_lpo(fold: SplitMasks, mudataset: MuDataLike, *, fold_index: int) -> None:
    """LPO pair disjointness is enforced by :func:`_validate_partition`."""


def _has_axis_overlap(fold: SplitMasks, *, axis: int) -> bool:
    """Whether an active row or column occurs in more than one split mask."""
    active = [mask.mask.any(axis=axis) for mask in (fold.train, fold.val, fold.test)]
    return bool(((active[0] & active[1]) | (active[0] & active[2]) | (active[1] & active[2])).any())


_VALIDATORS = {
    "LCO": _validate_lco,
    "LDO": _validate_ldo,
    "LTO": _validate_lto,
    "LPO": _validate_lpo,
}
