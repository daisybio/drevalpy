"""Tests for drevalpy.datasets.splits.manifest."""

from __future__ import annotations

import json
from pathlib import Path

from drevalpy.datasets.splits import (
    MANIFEST_FILENAME,
    SplitParams,
    read_manifest_test_mode,
    read_split_manifest,
    write_split_manifest,
)


def _sample_params(**overrides: object) -> SplitParams:
    defaults = {
        "test_mode": "LCO",
        "n_cv_splits": 2,
        "validation_ratio": 0.1,
        "random_state": 42,
        "split_early_stopping": True,
    }
    defaults.update(overrides)
    return SplitParams(**defaults)  # type: ignore[arg-type]


def test_write_split_manifest(tmp_path: Path) -> None:
    """
    Write split metadata to split_manifest.json.

    :param tmp_path: Temporary path provided by pytest.
    """
    params = _sample_params()
    write_split_manifest(
        tmp_path,
        params=params,
        split_label="scaling-lco",
        splits=[{"split_index": 0, "fraction": 0.5}],
    )
    payload = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert payload["split_label"] == "scaling-lco"
    assert payload["test_mode"] == "LCO"
    assert payload["n_cv_splits"] == 1
    assert payload["splits"][0]["fraction"] == 0.5
    assert "test_mode" not in payload["splits"][0]


def test_write_split_manifest_supports_nested_metadata(tmp_path: Path) -> None:
    """
    Persist nested metadata from external split scripts.

    :param tmp_path: Temporary path provided by pytest.
    """
    params = _sample_params()
    write_split_manifest(
        tmp_path,
        params=params,
        split_label="scaling-lco",
        splits=[
            {
                "split_index": 0,
                "groups": {"train": ["CL-0"], "validation": ["CL-1"], "test": ["CL-2"]},
            }
        ],
    )
    payload = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert payload["splits"][0]["groups"]["train"] == ["CL-0"]


def test_write_split_manifest_writes_test_mode_when_metadata_empty(tmp_path: Path) -> None:
    """
    Write a minimal manifest when no split metadata is provided.

    :param tmp_path: Temporary path provided by pytest.
    """
    params = _sample_params()
    write_split_manifest(tmp_path, params=params, split_label="LCO", splits=[])
    manifest_path = tmp_path / MANIFEST_FILENAME
    assert manifest_path.is_file()
    assert read_manifest_test_mode(manifest_path) == "LCO"
    payload = read_split_manifest(manifest_path)
    assert payload is not None
    assert payload["splits"] == []


def test_read_split_manifest_returns_none_for_missing_file(tmp_path: Path) -> None:
    """
    Return ``None`` when the manifest file does not exist.

    :param tmp_path: Temporary path provided by pytest.
    """
    assert read_split_manifest(tmp_path / MANIFEST_FILENAME) is None
