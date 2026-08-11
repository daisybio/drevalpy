"""Tests for external zoo YAML loading."""

from __future__ import annotations

import pytest

from drevalpy.models.zoo._external_load import (
    _collect_zoo_entries_from_yaml,
    _load_zoo_yaml_mapping,
)


def test_load_zoo_yaml_mapping_requires_file(tmp_path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match="not found"):
        _load_zoo_yaml_mapping(missing)


def test_collect_zoo_entries_rejects_non_mapping_entry(tmp_path) -> None:
    data = {"bad": "not-a-dict"}
    with pytest.raises(ValueError, match="must be a mapping"):
        _collect_zoo_entries_from_yaml(data, source=tmp_path / "z.yaml", builtin_names=frozenset())


def test_collect_zoo_entries_single_document_format(tmp_path) -> None:
    payload = {
        "predictor": "elasticNet",
        "cell_line_featurizer": "scaledGeneExpression",
        "drug_featurizer": "fingerprints",
        "name": "customEntry",
    }
    parsed = _collect_zoo_entries_from_yaml(payload, source=tmp_path / "z.yaml", builtin_names=frozenset())
    assert len(parsed) == 1
    assert parsed[0][0] == "customEntry"
    assert parsed[0][1].predictor.name == "elasticNet"
