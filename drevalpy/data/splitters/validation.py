"""Split validation: ensures folds satisfy their declared leakage constraints."""

from __future__ import annotations

from typing import Literal

import numpy as np

from drevalpy.data.structures import MuDataLike, SplitMasks

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
    for i, fold in enumerate(folds):
        validator(fold, mudataset, fold_index=i)


def _validate_lco(fold: SplitMasks, mudataset: MuDataLike, *, fold_index: int) -> None:
    """LCO: no cell line row has True in both train and test."""
    train_rows = np.where(fold.train.mask.any(axis=1))[0]
    test_rows = np.where(fold.test.mask.any(axis=1))[0]
    overlap = np.intersect1d(train_rows, test_rows)
    if len(overlap) > 0:
        raise SplitValidationError(
            f"LCO validation failed (fold {fold_index}): "
            f"{len(overlap)} cell line indices appear in both train and test."
        )


def _validate_ldo(fold: SplitMasks, mudataset: MuDataLike, *, fold_index: int) -> None:
    """LDO: no drug column has True in both train and test."""
    train_cols = np.where(fold.train.mask.any(axis=0))[0]
    test_cols = np.where(fold.test.mask.any(axis=0))[0]
    overlap = np.intersect1d(train_cols, test_cols)
    if len(overlap) > 0:
        raise SplitValidationError(
            f"LDO validation failed (fold {fold_index}): {len(overlap)} drug indices appear in both train and test."
        )


def _validate_lto(fold: SplitMasks, mudataset: MuDataLike, *, fold_index: int) -> None:
    """LTO: no tissue appears in both train and test cell lines."""
    cl_ids = mudataset.cell_line_ids
    tissues = mudataset.get_tissue(cl_ids)

    train_rows = np.where(fold.train.mask.any(axis=1))[0]
    test_rows = np.where(fold.test.mask.any(axis=1))[0]

    train_tissues = set(tissues[train_rows].tolist())
    test_tissues = set(tissues[test_rows].tolist())
    overlap = train_tissues & test_tissues
    if overlap:
        raise SplitValidationError(
            f"LTO validation failed (fold {fold_index}): "
            f"{len(overlap)} tissues appear in both train and test: {sorted(overlap)[:5]}"
        )


def _validate_lpo(fold: SplitMasks, mudataset: MuDataLike, *, fold_index: int) -> None:
    """LPO: no position is True in both train and test."""
    overlap_count = len(fold.train & fold.test)
    if overlap_count > 0:
        raise SplitValidationError(
            f"LPO validation failed (fold {fold_index}): "
            f"{overlap_count} (cell_line, drug) pairs appear in both train and test."
        )


_VALIDATORS = {
    "LCO": _validate_lco,
    "LDO": _validate_ldo,
    "LTO": _validate_lto,
    "LPO": _validate_lpo,
}
