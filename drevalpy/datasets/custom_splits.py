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
    """
    Compatibility wrapper for external split scripts.

    :param response_data: full response dataset passed to the splitter
    :param splitter: callable or path to a script defining ``create_splits``
    :param test_mode: one of ``LPO``, ``LCO``, ``LDO``, or ``LTO``
    :param n_cv_splits: requested number of CV splits from the pipeline
    :param validation_ratio: validation fraction from the pipeline
    :param random_state: random seed from the pipeline
    :param split_early_stopping: whether to derive early-stopping roles when absent
    :returns: validated splits and per-split metadata rows
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
    """
    Compatibility alias for ``create_splits`` using legacy argument names.

    :param response_data: full response dataset passed to the splitter
    :param custom_splitter: optional callable or script path defining ``create_splits``
    :param test_mode: one of ``LPO``, ``LCO``, ``LDO``, or ``LTO``; required when ``params`` is omitted
    :param n_cv_splits: requested number of CV splits from the pipeline
    :param validation_ratio: validation fraction from the pipeline
    :param random_state: random seed from the pipeline
    :param split_early_stopping: whether to derive early-stopping roles when absent
    :param params: optional pre-built split settings; overrides individual keyword args
    :returns: validated splits and per-split metadata rows
    :raises ValueError: if neither ``params`` nor ``test_mode`` is provided
    """
    if params is None and test_mode is None:
        msg = "Either params or test_mode must be provided"
        raise ValueError(msg)
    return create_splits(
        response_data,
        test_mode=params.test_mode if params is not None else test_mode,  # type: ignore[arg-type]
        external_splitter=custom_splitter,
        n_cv_splits=params.n_cv_splits if params is not None else n_cv_splits,
        validation_ratio=params.validation_ratio if params is not None else validation_ratio,
        random_state=params.random_state if params is not None else random_state,
        split_early_stopping=params.split_early_stopping if params is not None else split_early_stopping,
        params=params,
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
