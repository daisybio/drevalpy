"""Tests for the dataset registry and loader."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest

from drevalpy.registry.dataset import DatasetRegistry as Registry
from drevalpy.registry.dataset import DrevalConfig, SourceEntry, get_config_path
from drevalpy.registry.dataset import dataset_registry as registry

_EXPECTED_BUILTIN = [
    "BeatAML2",
    "CCLE",
    "CTRPv1",
    "CTRPv2",
    "GDSC1",
    "GDSC2",
    "PDX_Bruna",
]


class TestBuiltinRegistry:
    """Tests for the built-in dataset registry (packaged JSON)."""

    def test_available_datasets_json_structure(self) -> None:
        """Verify the packaged JSON has the expected schema."""
        registry_path = resources.files("drevalpy.data.datasets").joinpath("available_datasets.json")
        with registry_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)

        assert "sources" in raw
        assert "datasets" in raw

        for _source_name, val in raw["sources"].items():
            assert isinstance(val, (str, dict))

        for _ds_name, entry in raw["datasets"].items():
            assert "source" in entry
            assert "file" in entry
            assert entry["source"] in raw["sources"]
            assert entry["file"].endswith(".h5mu")

    def test_dataset_names(self) -> None:
        assert registry.dataset_names == sorted(_EXPECTED_BUILTIN)

    def test_source_names(self) -> None:
        assert "zenodo" in registry.source_names

    def test_is_registered(self) -> None:
        assert registry.is_registered("GDSC1")
        assert registry.is_registered("BeatAML2")
        assert not registry.is_registered("NonExistent")

    def test_builtin_vs_custom_separation(self) -> None:
        assert len(registry.builtin_datasets) == len(_EXPECTED_BUILTIN)
        assert len(registry.builtin_sources) >= 1

    def test_datasets_property_returns_merged(self) -> None:
        datasets = registry.datasets
        assert all(name in datasets for name in _EXPECTED_BUILTIN)

    def test_sources_have_urls(self) -> None:
        for source in registry.sources.values():
            assert source.url
            assert isinstance(source.url, str)


class TestRegistration:
    """Tests for register/unregister with a temporary config directory."""

    @pytest.fixture(autouse=True)
    def _use_tmp_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DREVALPY_CONFIG_DIR", str(tmp_path))
        self.reg = Registry()

    def test_register_source(self) -> None:
        self.reg.register_source("test_src", "https://example.com/data")
        assert "test_src" in self.reg.sources
        assert self.reg.sources["test_src"].url == "https://example.com/data"

    def test_register_source_with_storage_options(self) -> None:
        self.reg.register_source("s3_src", "s3://bucket/path", storage_options={"profile": "dev"})
        assert self.reg.sources["s3_src"].storage_options == {"profile": "dev"}

    def test_register_dataset(self) -> None:
        self.reg.register_source("src", "https://example.com")
        self.reg.register_dataset("MyData", source="src", file="MyData.h5mu")
        assert self.reg.is_registered("MyData")
        assert self.reg.datasets["MyData"].file == "MyData.h5mu"

    def test_register_dataset_unknown_source_raises(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            self.reg.register_dataset("MyData", source="nonexistent", file="x.h5mu")

    def test_unregister_dataset(self) -> None:
        self.reg.register_source("src", "https://example.com")
        self.reg.register_dataset("MyData", source="src", file="MyData.h5mu")
        self.reg.unregister_dataset("MyData")
        assert not self.reg.is_registered("MyData")

    def test_unregister_dataset_not_custom_raises(self) -> None:
        with pytest.raises(KeyError, match="not in custom"):
            self.reg.unregister_dataset("NonExistent")

    def test_unregister_source(self) -> None:
        self.reg.register_source("src", "https://example.com")
        self.reg.unregister_source("src")
        assert "src" not in self.reg.custom_sources

    def test_unregister_source_with_datasets_raises(self) -> None:
        self.reg.register_source("src", "https://example.com")
        self.reg.register_dataset("MyData", source="src", file="x.h5mu")
        with pytest.raises(ValueError, match="still referenced"):
            self.reg.unregister_source("src")

    def test_unregister_source_not_custom_raises(self) -> None:
        with pytest.raises(KeyError, match="not in custom"):
            self.reg.unregister_source("NonExistent")

    def test_custom_overrides_builtin(self) -> None:
        self.reg.register_source("zenodo", "s3://my-mirror/data")
        assert self.reg.sources["zenodo"].url == "s3://my-mirror/data"


class TestPersistence:
    """Tests for config persistence and reload."""

    @pytest.fixture(autouse=True)
    def _use_tmp_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DREVALPY_CONFIG_DIR", str(tmp_path))
        self.tmp_path = tmp_path
        self.reg = Registry()

    def test_registration_persists_to_disk(self) -> None:
        self.reg.register_source("persisted", "https://example.com")
        config_path = get_config_path()
        assert config_path.is_file()

        with open(config_path) as f:
            raw = json.load(f)
        assert "persisted" in raw["sources"]

    def test_reload_picks_up_external_changes(self) -> None:
        config_path = get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(
                {
                    "sources": {"external": "https://external.org"},
                    "datasets": {"ExtData": {"source": "external", "file": "ext.h5mu"}},
                }
            )
        )

        self.reg.reload()
        assert self.reg.is_registered("ExtData")
        assert "external" in self.reg.custom_sources

    def test_atomic_write_preserves_existing_entries(self) -> None:
        self.reg.register_source("first", "https://first.com")
        self.reg.register_source("second", "https://second.com")
        assert "first" in self.reg.custom_sources
        assert "second" in self.reg.custom_sources


class TestModels:
    """Tests for Pydantic config models."""

    def test_source_entry_from_string(self) -> None:
        entry = SourceEntry.from_raw("https://example.com")
        assert entry.url == "https://example.com"
        assert entry.storage_options == {}

    def test_source_entry_from_dict(self) -> None:
        entry = SourceEntry.from_raw({"url": "s3://bucket", "storage_options": {"anon": True}})
        assert entry.url == "s3://bucket"
        assert entry.storage_options == {"anon": True}

    def test_source_entry_to_raw_simple(self) -> None:
        entry = SourceEntry(url="https://example.com")
        assert entry.to_raw() == "https://example.com"

    def test_source_entry_to_raw_with_options(self) -> None:
        entry = SourceEntry(url="s3://bucket", storage_options={"profile": "dev"})
        assert entry.to_raw() == {"url": "s3://bucket", "storage_options": {"profile": "dev"}}

    def test_dreval_config_roundtrip(self) -> None:
        raw = {
            "sources": {"lab": "https://lab.org/data"},
            "datasets": {"Study1": {"source": "lab", "file": "Study1.h5mu"}},
        }
        config = DrevalConfig.from_raw(raw)
        assert "lab" in config.sources
        assert "Study1" in config.datasets
        assert config.to_raw() == raw

    def test_dreval_config_forbids_extra_keys(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DrevalConfig(
                sources={},
                datasets={},
                unknown="bad",
            )


class TestStrRepr:
    """Tests for string representation."""

    def test_str_contains_dataset_names(self) -> None:
        output = str(registry)
        assert "GDSC1" in output
        assert "BeatAML2" in output

    def test_str_contains_source_names(self) -> None:
        output = str(registry)
        assert "zenodo" in output

    def test_repr_equals_str(self) -> None:
        assert str(registry) == repr(registry)
