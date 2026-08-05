"""Tests for the built-in dataset registry and loader."""

import json
from importlib import resources
from unittest.mock import patch

import pandas as pd
import pytest

from drevalpy.datasets.loader import (
    BuiltinDatasetEntry,
    _load_builtin,
    is_builtin_dataset,
    list_builtin_datasets,
    load_dataset,
    load_response_dataset,
)

_REGISTRY_JSON = "available_datasets.json"
_KNOWN_SOURCE_KINDS = frozenset({"zenodo", "nfcore_test"})
_EXPECTED_BUILTIN = [
    "BeatAML2",
    "CCLE",
    "CTRPv1",
    "CTRPv2",
    "GDSC1",
    "GDSC2",
    "PDX_Bruna",
    "TOYv1",
    "TOYv2",
]


def _load_registry_json() -> dict:
    registry_path = resources.files("drevalpy.datasets").joinpath(_REGISTRY_JSON)
    with registry_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_available_datasets_json_structure() -> None:
    raw = _load_registry_json()
    assert raw["default_measure"] == "LN_IC50_curvecurator"

    sources = raw["sources"]
    datasets = raw["datasets"]
    assert datasets

    names = [entry["name"] for entry in datasets]
    assert len(names) == len(set(names)), "duplicate dataset names"
    assert sorted(names) == sorted(_EXPECTED_BUILTIN)

    for entry in datasets:
        assert entry["name"]
        assert entry["source"] in sources
        assert entry["response_file"]
        assert entry["response_file"] == f"{entry['name']}/{entry['name']}.csv"

    for cfg in sources.values():
        assert cfg["kind"] in _KNOWN_SOURCE_KINDS
        if cfg["kind"] == "nfcore_test":
            assert cfg.get("base_url")


def test_list_builtin_datasets_matches_registry() -> None:
    assert list_builtin_datasets() == sorted(_EXPECTED_BUILTIN)
    assert len(list_builtin_datasets()) == 9


def test_is_builtin_dataset() -> None:
    assert is_builtin_dataset("TOYv1")
    assert is_builtin_dataset("GDSC1")
    assert not is_builtin_dataset("MyStudy")


def test_load_builtin_applies_tissue_override(tmp_path) -> None:
    entry = BuiltinDatasetEntry(
        name="BeatAML2",
        source="zenodo",
        response_file="BeatAML2/BeatAML2.csv",
        tissue_override="Blood",
    )
    csv_path = tmp_path / "BeatAML2" / "BeatAML2.csv"
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "pubchem_id": ["1"],
            "cell_line_name": ["A"],
            "tissue": ["Other"],
            "LN_IC50_curvecurator": [1.0],
        }
    ).to_csv(csv_path, index=False)

    with patch("drevalpy.datasets.loader._ensure_builtin_artifacts"):
        dataset = _load_builtin(entry, str(tmp_path), measure="LN_IC50_curvecurator")

    tissue = dataset.tissue
    assert tissue is not None
    assert tissue[0] == "Blood"


def test_load_builtin_rejects_unknown_measure(tmp_path) -> None:
    entry = BuiltinDatasetEntry(name="TOYv1", source="zenodo", response_file="TOYv1/TOYv1.csv")
    csv_path = tmp_path / "TOYv1" / "TOYv1.csv"
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "pubchem_id": ["1"],
            "cell_line_name": ["A"],
            "tissue": ["Lung"],
            "LN_IC50_curvecurator": [1.0],
        }
    ).to_csv(csv_path, index=False)

    with (
        patch("drevalpy.datasets.loader._ensure_builtin_artifacts"),
        pytest.raises(ValueError, match="Measure 'missing_measure'"),
    ):
        _load_builtin(entry, str(tmp_path), measure="missing_measure")


def test_load_dataset_alias_matches_load_response_dataset(tmp_path) -> None:
    dataset_name = "AliasStudy"
    csv_path = tmp_path / dataset_name / f"{dataset_name}.csv"
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "cell_line_name": ["A"],
            "pubchem_id": ["D"],
            "response": [0.5],
        }
    ).to_csv(csv_path, index=False)

    via_alias = load_dataset(dataset_name, path_data=str(tmp_path))  # noqa: S615  # public alias parity
    via_primary = load_response_dataset(dataset_name, path_data=str(tmp_path))
    assert via_alias.response == via_primary.response
    assert via_alias.cell_line_ids == via_primary.cell_line_ids
    assert via_alias.drug_ids == via_primary.drug_ids


def test_load_dataset_custom(tmp_path) -> None:
    dataset_name = "CustomStudy"
    csv_path = tmp_path / dataset_name / f"{dataset_name}.csv"
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "cell_line_name": ["A"],
            "pubchem_id": ["D"],
            "response": [0.5],
        }
    ).to_csv(csv_path, index=False)

    dataset = load_response_dataset(dataset_name, path_data=str(tmp_path))
    assert len(dataset.response) == 1
