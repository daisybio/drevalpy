"""Validation helpers for built-in and external split providers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...pipeline_function import pipeline_function
from ..dataset import DrugResponseDataset, split_early_stopping_data
from .types import OPTIONAL_ROLES, REQUIRED_ROLES, TEST_MODES, SplitError


def _row_keys(dataset: DrugResponseDataset) -> set[tuple[str, str, float]]:
    """
    Build exact row identifiers for overlap checks.

    :param dataset: response dataset whose rows are keyed
    :returns: set of ``(cell_line_id, drug_id, response)`` tuples
    """
    return {
        (str(cl), str(drug), float(resp))
        for cl, drug, resp in zip(dataset.cell_line_ids, dataset.drug_ids, dataset.response, strict=True)
    }


def _group_ids(dataset: DrugResponseDataset, test_mode: str) -> set[Any]:
    """
    Extract leave-out group identifiers for a ``test_mode``.

    :param dataset: response dataset whose groups are extracted
    :param test_mode: one of ``LPO``, ``LCO``, ``LDO``, or ``LTO``
    :returns: cell-line, drug, tissue, or pair identifiers depending on ``test_mode``
    :raises SplitError: if ``test_mode`` is unknown or LTO tissue is missing
    """
    if test_mode == "LCO":
        return set(map(str, dataset.cell_line_ids))
    if test_mode == "LDO":
        return set(map(str, dataset.drug_ids))
    if test_mode == "LTO":
        if dataset.tissue is None:
            msg = "LTO validation requires tissue annotations on all role datasets"
            raise SplitError(msg)
        return set(map(str, dataset.tissue))
    if test_mode == "LPO":
        return {(str(cl), str(drug)) for cl, drug in zip(dataset.cell_line_ids, dataset.drug_ids, strict=True)}
    msg = f"Unknown test_mode {test_mode!r}; choose from {sorted(TEST_MODES)}"
    raise SplitError(msg)


def _assert_disjoint_groups(
    roles: dict[str, DrugResponseDataset],
    test_mode: str,
    *,
    split_index: int,
) -> None:
    """
    Ensure train, validation, and test do not share leave-out groups.

    :param roles: split role datasets to check
    :param test_mode: one of ``LPO``, ``LCO``, ``LDO``, or ``LTO``
    :param split_index: index of the split within the returned list
    :raises SplitError: if the same group appears in multiple roles
    """
    seen: dict[Any, str] = {}
    for role in REQUIRED_ROLES:
        for group in _group_ids(roles[role], test_mode):
            if group in seen:
                msg = (
                    f"Split {split_index}: {test_mode} leakage — " f"{seen[group]!r} and {role!r} share group {group!r}"
                )
                raise SplitError(msg)
            seen[group] = role


def _assert_disjoint_rows(
    roles: dict[str, DrugResponseDataset],
    *,
    split_index: int,
) -> None:
    """
    Ensure train, validation, and test do not share exact response rows.

    :param roles: split role datasets to check
    :param split_index: index of the split within the returned list
    :raises SplitError: if the same row appears in multiple roles
    """
    seen: dict[tuple[str, str, float], str] = {}
    for role in REQUIRED_ROLES:
        for row in _row_keys(roles[role]):
            if row in seen:
                msg = f"Split {split_index}: exact row overlap between " f"{seen[row]!r} and {role!r}: {row[:2]}"
                raise SplitError(msg)
            seen[row] = role


def _normalize_split_dict(
    raw: dict[str, Any], *, split_index: int
) -> tuple[dict[str, DrugResponseDataset], dict[str, Any]]:
    """
    Parse and validate one raw split dict from a split provider.

    :param raw: split dict returned by a provider
    :param split_index: index of the split within the returned list
    :returns: validated role datasets and optional metadata dict
    :raises SplitError: if roles are missing, empty, or have wrong types
    """
    metadata = raw.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        msg = f"Split {split_index}: metadata must be a dict when provided"
        raise SplitError(msg)

    roles: dict[str, DrugResponseDataset] = {}
    for role in REQUIRED_ROLES + OPTIONAL_ROLES:
        if role not in raw:
            continue
        value = raw[role]
        if not isinstance(value, DrugResponseDataset):
            msg = f"Split {split_index}: {role!r} must be a DrugResponseDataset, got {type(value).__name__}"
            raise SplitError(msg)
        roles[role] = value

    for role in REQUIRED_ROLES:
        if role not in roles:
            msg = f"Split {split_index}: missing required role {role!r}"
            raise SplitError(msg)
        if len(roles[role]) == 0:
            msg = f"Split {split_index}: role {role!r} must not be empty"
            raise SplitError(msg)

    dataset_names = {roles[role].dataset_name for role in REQUIRED_ROLES}
    if len(dataset_names) != 1:
        msg = f"Split {split_index}: inconsistent dataset_name across roles: {dataset_names}"
        raise SplitError(msg)

    return roles, metadata if isinstance(metadata, dict) else {}


@pipeline_function
def validate_splits(
    splits: list[dict[str, Any]],
    test_mode: str,
) -> tuple[list[dict[str, DrugResponseDataset]], list[dict[str, Any]]]:
    """
    Validate split output according to ``test_mode`` semantics.

    :param splits: raw split dicts returned by a built-in or external provider
    :param test_mode: one of ``LPO``, ``LCO``, ``LDO``, or ``LTO``
    :returns: validated role datasets and optional per-split metadata rows
    :raises SplitError: if splits are missing roles or leak across groups/rows
    """
    if test_mode not in TEST_MODES:
        msg = f"Unknown test_mode {test_mode!r}; choose from {sorted(TEST_MODES)}"
        raise SplitError(msg)
    if not splits:
        msg = "Split provider returned no splits"
        raise SplitError(msg)

    validated: list[dict[str, DrugResponseDataset]] = []
    metadata_rows: list[dict[str, Any]] = []
    for split_index, raw in enumerate(splits):
        if not isinstance(raw, dict):
            msg = f"Split {split_index}: expected dict, got {type(raw).__name__}"
            raise SplitError(msg)
        roles, metadata = _normalize_split_dict(raw, split_index=split_index)
        _assert_disjoint_rows(roles, split_index=split_index)
        _assert_disjoint_groups(roles, test_mode, split_index=split_index)
        validated.append(roles)
        metadata_rows.append({"split_index": split_index, **metadata})

    return validated, metadata_rows


@pipeline_function
def ensure_early_stopping_splits(
    splits: list[dict[str, DrugResponseDataset]],
    test_mode: str,
) -> None:
    """
    Fill ``validation_es`` and ``early_stopping`` when absent.

    :param splits: validated split dicts to mutate in place
    :param test_mode: one of ``LPO``, ``LCO``, ``LDO``, or ``LTO``
    """
    for split in splits:
        if "validation_es" in split and "early_stopping" in split:
            continue
        validation = split["validation"]
        n_groups = len(_group_ids(validation, test_mode))
        if n_groups < 2:
            split["validation_es"] = validation.copy()
            split["early_stopping"] = DrugResponseDataset(
                response=np.array([]),
                cell_line_ids=np.array([]),
                drug_ids=np.array([]),
                tissues=np.array([]) if validation.tissue is not None else None,
                dataset_name=validation.dataset_name,
            )
            continue
        validation_es, early_stopping = split_early_stopping_data(validation, test_mode=test_mode)
        split["validation_es"] = validation_es
        split["early_stopping"] = early_stopping
