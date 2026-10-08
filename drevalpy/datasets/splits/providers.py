"""Built-in and external split providers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from ...pipeline_function import pipeline_function
from ..dataset import DrugResponseDataset
from .manifest import write_split_manifest
from .types import (
    TEST_MODES,
    ExternalSplitCreator,
    SplitError,
    SplitParams,
    SplitResult,
    make_split_params,
)
from .validation import ensure_early_stopping_splits, validate_splits


def load_external_splitter(path: Path | str) -> ExternalSplitCreator:
    """
    Load a module-level ``create_splits`` function from a Python script.

    :param path: path to a Python file defining ``create_splits(response_data, params)``
    :returns: the loaded splitter callable
    :raises FileNotFoundError: if the script path does not exist
    :raises ImportError: if the script cannot be imported
    :raises AttributeError: if ``create_splits`` is missing
    :raises TypeError: if ``create_splits`` is not callable
    """
    script_path = Path(path).expanduser().resolve()
    if not script_path.is_file():
        msg = f"External split script not found: {script_path}"
        raise FileNotFoundError(msg)

    spec = importlib.util.spec_from_file_location("_drevalpy_external_split_", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load external split script: {script_path}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "create_splits", None)
    if fn is None:
        msg = f"{script_path} must define a module-level function create_splits(response_data, params)"
        raise AttributeError(msg)
    if not callable(fn):
        msg = "create_splits must be callable"
        raise TypeError(msg)
    return fn  # type: ignore[return-value]


def _raw_builtin_splits(response_data: DrugResponseDataset, params: SplitParams) -> list[dict[str, Any]]:
    """
    Generate raw fold dictionaries from built-in splitting logic.

    :param response_data: full response dataset to split
    :param params: pipeline split settings
    :returns: raw split dicts before shared validation
    :raises SplitError: if ``test_mode`` is unknown
    """
    if params.test_mode not in TEST_MODES:
        msg = f"Unknown test_mode {params.test_mode!r}; choose from {sorted(TEST_MODES)}"
        raise SplitError(msg)

    working = response_data.copy()
    return working.split_dataset(
        n_cv_splits=params.n_cv_splits,
        mode=params.test_mode,
        split_validation=True,
        validation_ratio=params.validation_ratio,
        random_state=params.random_state,
        split_early_stopping=False,
    )


def _finalize_splits(raw_splits: list[dict[str, Any]], params: SplitParams) -> SplitResult:
    """
    Normalize and validate split output from any provider.

    :param raw_splits: fold dictionaries returned by a provider
    :param params: pipeline split settings
    :returns: validated splits and per-split metadata rows
    """
    validated, metadata_rows = validate_splits(raw_splits, params.test_mode)
    if params.split_early_stopping:
        ensure_early_stopping_splits(validated, params.test_mode)
    return validated, metadata_rows


@pipeline_function
def run_builtin_splitter(response_data: DrugResponseDataset, params: SplitParams) -> SplitResult:
    """
    Create built-in CV splits using the shared validation path.

    :param response_data: full response dataset to split
    :param params: pipeline split settings
    :returns: validated splits and per-split metadata rows
    """
    validated = _raw_builtin_splits(response_data, params)
    metadata_rows = [{"split_index": split_index} for split_index in range(len(validated))]
    if params.split_early_stopping:
        ensure_early_stopping_splits(validated, params.test_mode)
    return validated, metadata_rows


@pipeline_function
def run_external_splitter(
    response_data: DrugResponseDataset,
    splitter: ExternalSplitCreator | str | Path,
    params: SplitParams,
) -> SplitResult:
    """
    Execute an external splitter and validate its output.

    :param response_data: full response dataset passed to the splitter
    :param splitter: callable or path to a script defining ``create_splits``
    :param params: pipeline split settings
    :returns: validated splits and per-split metadata rows
    """
    if isinstance(splitter, (str, Path)):
        splitter = load_external_splitter(splitter)
    raw_splits = splitter(response_data.copy(), params)
    return _finalize_splits(raw_splits, params)


@pipeline_function
def create_splits(
    response_data: DrugResponseDataset,
    *,
    test_mode: str | None = None,
    external_splitter: ExternalSplitCreator | str | Path | None = None,
    n_cv_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    split_early_stopping: bool = True,
    params: SplitParams | None = None,
) -> SplitResult:
    """
    Create CV splits for a required ``test_mode`` via built-in or external providers.

    :param response_data: full response dataset passed to the splitter
    :param test_mode: one of ``LPO``, ``LCO``, ``LDO``, or ``LTO``; required when ``params`` is omitted
    :param external_splitter: optional callable or script path defining ``create_splits``
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
    split_params = params or make_split_params(
        test_mode=test_mode,  # type: ignore[arg-type]
        n_cv_splits=n_cv_splits,
        validation_ratio=validation_ratio,
        random_state=random_state,
        split_early_stopping=split_early_stopping,
    )
    if external_splitter is not None:
        return run_external_splitter(response_data, external_splitter, split_params)
    return run_builtin_splitter(response_data, split_params)


@pipeline_function
def create_and_record_splits(
    response_data: DrugResponseDataset,
    *,
    split_path: Path | str,
    split_label: str,
    external_splitter: ExternalSplitCreator | str | Path | None = None,
    test_mode: str | None = None,
    n_cv_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    split_early_stopping: bool = True,
    params: SplitParams | None = None,
) -> SplitResult:
    """
    Create CV splits, attach them to the dataset, and write the split manifest.

    :param response_data: full response dataset to split
    :param split_path: directory where the split manifest is written
    :param split_label: result-directory label recorded in the manifest
    :param external_splitter: optional callable or script path defining ``create_splits``
    :param test_mode: one of ``LPO``, ``LCO``, ``LDO``, or ``LTO``; required when ``params`` is omitted
    :param n_cv_splits: requested number of CV splits from the pipeline
    :param validation_ratio: validation fraction from the pipeline
    :param random_state: random seed from the pipeline
    :param split_early_stopping: whether to derive early-stopping roles when absent
    :param params: optional pre-built split settings; overrides individual keyword args
    :returns: validated splits and per-split metadata rows
    """
    response_data.remove_nan_responses()
    split_params = params or make_split_params(
        test_mode=test_mode,  # type: ignore[arg-type]
        n_cv_splits=n_cv_splits,
        validation_ratio=validation_ratio,
        random_state=random_state,
        split_early_stopping=split_early_stopping,
    )
    cv_splits, metadata_rows = create_splits(
        response_data,
        params=split_params,
        external_splitter=external_splitter,
    )
    response_data._cv_splits = cv_splits
    write_split_manifest(
        split_path,
        params=split_params,
        split_label=split_label,
        splits=metadata_rows,
    )
    return cv_splits, metadata_rows
