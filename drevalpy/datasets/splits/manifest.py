"""Split manifest read/write helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import SplitError, SplitParams

MANIFEST_FILENAME = "split_manifest.json"


def validate_split_label(label: str) -> str:
    """
    Ensure a result-directory label is safe for paths and report parsing.

    :param label: directory name used under the dataset results folder
    :returns: the validated label unchanged
    :raises SplitError: if the label is empty or contains path separators
    """
    if not label or label.strip() != label:
        msg = "split label must be a non-empty string without leading or trailing whitespace"
        raise SplitError(msg)
    if "/" in label or "\\" in label:
        msg = f"split label must not contain path separators: {label!r}"
        raise SplitError(msg)
    return label


def build_split_manifest(
    params: SplitParams,
    split_label: str,
    splits: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the JSON-serializable split manifest payload.

    Run-level settings live at the top level; per-fold metadata stays under ``splits``.

    :param params: pipeline split settings
    :param split_label: directory label used under the dataset results folder
    :param splits: per-split metadata rows
    :returns: manifest payload ready for JSON encoding
    """
    return {
        "split_label": split_label,
        "test_mode": params.test_mode,
        "n_cv_splits": len(splits),
        "validation_ratio": params.validation_ratio,
        "random_state": params.random_state,
        "split_early_stopping": params.split_early_stopping,
        "splits": splits,
    }


def write_split_manifest(
    path: Path | str,
    *,
    params: SplitParams,
    split_label: str,
    splits: list[dict[str, Any]],
) -> None:
    """
    Write split metadata next to persisted split files.

    :param path: directory where split CSV files are stored
    :param params: pipeline split settings recorded in the manifest
    :param split_label: directory label used under the dataset results folder
    :param splits: per-split metadata collected during validation
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    payload = build_split_manifest(params, split_label, splits)
    manifest_path = out / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_split_manifest(manifest_path: Path | str) -> dict[str, Any] | None:
    """
    Read a split manifest file.

    :param manifest_path: path to ``split_manifest.json``
    :returns: parsed manifest payload, or ``None`` when absent or invalid
    """
    path = Path(manifest_path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def read_manifest_test_mode(manifest_path: Path | str) -> str | None:
    """
    Read the semantic ``test_mode`` from a split manifest file.

    :param manifest_path: path to ``split_manifest.json``
    :returns: top-level ``test_mode`` value, or ``None`` when absent
    """
    payload = read_split_manifest(manifest_path)
    if payload is None:
        return None
    value = payload.get("test_mode")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
