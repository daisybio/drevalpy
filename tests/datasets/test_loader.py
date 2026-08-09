"""Tests for the built-in dataset registry and loader."""

import json
from importlib import resources

from drevalpy.datasets import registry

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
    assert registry.list_datasets() == sorted(_EXPECTED_BUILTIN)
    assert len(registry.list_datasets()) == 9


def test_is_builtin_dataset() -> None:
    assert registry.is_registered("TOYv1")
    assert registry.is_registered("GDSC1")
    assert not registry.is_registered("MyStudy")
