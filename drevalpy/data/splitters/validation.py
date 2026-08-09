"""Split validation: ensures folds satisfy their declared leakage constraints."""

from __future__ import annotations

from typing import Literal

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
    """LCO: no cell line index appears in both train and test."""
    train = set(fold.train_cell_lines.tolist())
    test = set(fold.test_cell_lines.tolist())
    overlap = train & test
    if overlap:
        raise SplitValidationError(
            f"LCO validation failed (fold {fold_index}): "
            f"{len(overlap)} cell line indices appear in both train and test."
        )


def _validate_ldo(fold: SplitMasks, mudataset: MuDataLike, *, fold_index: int) -> None:
    """LDO: no drug index appears in both train and test."""
    if fold.train_drugs is None or fold.test_drugs is None:
        raise SplitValidationError(
            f"LDO validation failed (fold {fold_index}): drug indices are None."
        )
    train = set(fold.train_drugs.tolist())
    test = set(fold.test_drugs.tolist())
    overlap = train & test
    if overlap:
        raise SplitValidationError(
            f"LDO validation failed (fold {fold_index}): "
            f"{len(overlap)} drug indices appear in both train and test."
        )


def _validate_lto(fold: SplitMasks, mudataset: MuDataLike, *, fold_index: int) -> None:
    """LTO: no tissue appears in both train and test cell lines."""
    cl_ids = mudataset.cell_line_ids
    tissues = mudataset.get_tissue(cl_ids)

    train_tissues = set(tissues[fold.train_cell_lines].tolist())
    test_tissues = set(tissues[fold.test_cell_lines].tolist())
    overlap = train_tissues & test_tissues
    if overlap:
        raise SplitValidationError(
            f"LTO validation failed (fold {fold_index}): "
            f"{len(overlap)} tissues appear in both train and test: {sorted(overlap)[:5]}"
        )


def _validate_lpo(fold: SplitMasks, mudataset: MuDataLike, *, fold_index: int) -> None:
    """LPO: no (cell_line_idx, drug_idx) pair appears in both train and test."""
    if fold.train_drugs is None or fold.test_drugs is None:
        raise SplitValidationError(
            f"LPO validation failed (fold {fold_index}): drug indices are None."
        )
    train_pairs = set(zip(fold.train_cell_lines.tolist(), fold.train_drugs.tolist(), strict=True))
    test_pairs = set(zip(fold.test_cell_lines.tolist(), fold.test_drugs.tolist(), strict=True))
    overlap = train_pairs & test_pairs
    if overlap:
        raise SplitValidationError(
            f"LPO validation failed (fold {fold_index}): "
            f"{len(overlap)} (cell_line, drug) pairs appear in both train and test."
        )


_VALIDATORS = {
    "LCO": _validate_lco,
    "LDO": _validate_ldo,
    "LTO": _validate_lto,
    "LPO": _validate_lpo,
}
