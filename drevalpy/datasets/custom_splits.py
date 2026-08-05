"""Compatibility exports for the split provider package (issue #407)."""

from __future__ import annotations

from pathlib import Path

from .dataset import DrugResponseDataset
from .splits import (
    MANIFEST_FILENAME,
    OPTIONAL_ROLES,
    REQUIRED_ROLES,
    TEST_MODES,
    ExternalSplitCreator,
    SplitCreator,
    SplitError,
    SplitParams,
    SplitResult,
    create_splits,
    ensure_early_stopping_splits,
    load_external_splitter,
    read_manifest_test_mode,
    read_split_manifest,
    run_builtin_splitter,
    run_external_splitter,
    validate_split_label,
    validate_splits,
    write_split_manifest,
)

CustomSplitError = SplitError
CustomSplitParams = SplitParams
CustomSplitCreator = ExternalSplitCreator
load_custom_splitter = load_external_splitter
validate_cv_splits = validate_splits


def run_custom_splitter(
    response_data: DrugResponseDataset,
    splitter: ExternalSplitCreator | str | Path,
    *,
    test_mode: str,
    n_cv_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    split_early_stopping: bool = True,
) -> SplitResult:
    """Compatibility wrapper for external split scripts.

    :param response_data: Full response dataset passed to the splitter.
    :param splitter: Callable or path to a script defining ``create_splits``.
    :param test_mode: One of ``LPO``, ``LCO``, ``LDO``, or ``LTO``.
    :param n_cv_splits: Requested number of CV splits.
    :param validation_ratio: Validation fraction of the training set.
    :param random_state: Random seed for splitting.
    :param split_early_stopping: Whether to derive early-stopping roles when absent.
    :returns: Validated splits and per-split metadata rows.
    """
    return create_splits(
        response_data,
        test_mode=test_mode,
        external_splitter=splitter,
        n_cv_splits=n_cv_splits,
        validation_ratio=validation_ratio,
        random_state=random_state,
        split_early_stopping=split_early_stopping,
    )


def run_splitter(
    response_data: DrugResponseDataset,
    *,
    custom_splitter: ExternalSplitCreator | str | Path | None = None,
    test_mode: str | None = None,
    n_cv_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    split_early_stopping: bool = True,
    params: SplitParams | None = None,
) -> SplitResult:
    """Compatibility alias for ``create_splits`` using legacy argument names.

    :param response_data: Full response dataset passed to the splitter.
    :param custom_splitter: Optional callable or script path defining ``create_splits``.
    :param test_mode: One of ``LPO``, ``LCO``, ``LDO``, or ``LTO``; required when ``params`` is omitted.
    :param n_cv_splits: Requested number of CV splits.
    :param validation_ratio: Validation fraction of the training set.
    :param random_state: Random seed for splitting.
    :param split_early_stopping: Whether to derive early-stopping roles when absent.
    :param params: Pre-built split settings; overrides individual keyword args.
    :returns: Validated splits and per-split metadata rows.
    :raises ValueError: If neither ``params`` nor ``test_mode`` is provided.
    """
    if params is not None:
        return create_splits(
            response_data,
            params=params,
            external_splitter=custom_splitter,
        )
    if test_mode is None:
        msg = "Either params or test_mode must be provided"
        raise ValueError(msg)
    return create_splits(
        response_data,
        test_mode=test_mode,
        external_splitter=custom_splitter,
        n_cv_splits=n_cv_splits,
        validation_ratio=validation_ratio,
        random_state=random_state,
        split_early_stopping=split_early_stopping,
    )


__all__ = [
    "MANIFEST_FILENAME",
    "OPTIONAL_ROLES",
    "REQUIRED_ROLES",
    "TEST_MODES",
    "CustomSplitCreator",
    "CustomSplitError",
    "CustomSplitParams",
    "ExternalSplitCreator",
    "SplitCreator",
    "SplitError",
    "SplitParams",
    "create_splits",
    "ensure_early_stopping_splits",
    "load_custom_splitter",
    "load_external_splitter",
    "read_manifest_test_mode",
    "read_split_manifest",
    "run_builtin_splitter",
    "run_custom_splitter",
    "run_external_splitter",
    "run_splitter",
    "validate_cv_splits",
    "validate_split_label",
    "validate_splits",
    "write_split_manifest",
]
