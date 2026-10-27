"""Tests for :class:`~drevalpy.registry.dataset._registry.DatasetRegistry`.

Every test drives a fresh ``DatasetRegistry`` whose config lives in ``tmp_path``, so
neither the singleton nor the developer's real ``datasets.json`` is mutated.
"""

from __future__ import annotations

import json

import pytest
from upath import UPath

from drevalpy.errors import UnknownNameError
from drevalpy.registry import dataset as dataset_facade
from drevalpy.registry.dataset._models import DatasetEntry, SourceEntry
from drevalpy.registry.dataset._registry import DatasetRegistry, dataset_registry

_BUILTIN_DATASET = "GDSC1"
_BUILTIN_SOURCE = "orakl"


@pytest.fixture
def config_path(tmp_path, monkeypatch: pytest.MonkeyPatch) -> UPath:
    monkeypatch.setenv("DREVALPY_CONFIG_DIR", str(tmp_path))
    return UPath(tmp_path) / "datasets.json"


@pytest.fixture
def registry(config_path: UPath) -> DatasetRegistry:
    return DatasetRegistry()


def _saved(config_path: UPath, section: str) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))[section]


def test_builtin_entries_are_loaded_and_custom_entries_start_empty(registry: DatasetRegistry) -> None:
    entry = registry.builtin_datasets[_BUILTIN_DATASET]
    assert (entry.source, entry.file) == (_BUILTIN_SOURCE, f"{_BUILTIN_DATASET}.h5mu")
    assert registry.builtin_sources[_BUILTIN_SOURCE].url.startswith("s3://")
    assert (registry.custom_datasets, registry.custom_sources) == ({}, {})
    assert registry.dataset_names == sorted(registry.dataset_names)
    assert registry.source_names == sorted(registry.source_names)
    assert registry.is_registered(_BUILTIN_DATASET) is True
    assert registry.is_registered("NotADataset") is False


class TestRegisterSource:
    def test_persists_the_entry_and_keeps_storage_options(self, registry: DatasetRegistry, config_path: UPath) -> None:
        registry.register_source("local", "file:///data")
        registry.register_source("bucket", "s3://bucket/", {"anon": True})

        assert _saved(config_path, "sources")["local"] == "file:///data"
        assert registry.custom_sources["bucket"] == SourceEntry(url="s3://bucket/", storage_options={"anon": True})

    def test_custom_sources_override_builtins(self, registry: DatasetRegistry) -> None:
        registry.register_source(_BUILTIN_SOURCE, "file:///override", override=True)

        assert registry.sources[_BUILTIN_SOURCE].url == "file:///override"
        assert registry.builtin_sources[_BUILTIN_SOURCE].url != "file:///override"

    def test_identical_repeat_leaves_the_file_untouched(self, registry: DatasetRegistry, config_path: UPath) -> None:
        registry.register_source("local", "file:///data", {"anon": True})
        config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        written = config_path.read_text(encoding="utf-8")

        registry.register_source("local", "file:///data", {"anon": True})

        assert config_path.read_text(encoding="utf-8") == written

    def test_identical_to_a_builtin_is_a_noop(self, registry: DatasetRegistry, config_path: UPath) -> None:
        registry.register_source(_BUILTIN_SOURCE, registry.builtin_sources[_BUILTIN_SOURCE].url)

        assert registry.custom_sources == {}
        assert not config_path.exists()

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            (("file:///data", None), ("file:///elsewhere", None)),
            (("s3://bucket/", {"anon": True}), ("s3://bucket/", {"anon": False})),
        ],
        ids=["url", "storage-options"],
    )
    def test_conflict_raises_and_keeps_the_original(self, registry: DatasetRegistry, first, second) -> None:
        registry.register_source("local", *first)

        with pytest.raises(ValueError, match="Source 'local' already registered"):
            registry.register_source("local", *second)

        assert registry.custom_sources["local"].url == first[0]

    def test_conflict_with_a_builtin_raises(self, registry: DatasetRegistry, config_path: UPath) -> None:
        with pytest.raises(ValueError, match=f"Source '{_BUILTIN_SOURCE}' already registered"):
            registry.register_source(_BUILTIN_SOURCE, "file:///override")

        assert not config_path.exists()

    def test_override_replaces_and_persists(self, registry: DatasetRegistry, config_path: UPath) -> None:
        registry.register_source("local", "file:///data")

        registry.register_source("local", "file:///elsewhere", override=True)

        assert _saved(config_path, "sources") == {"local": "file:///elsewhere"}

    def test_sees_an_entry_written_by_another_process(self, registry: DatasetRegistry, config_path: UPath) -> None:
        config_path.write_text(json.dumps({"sources": {"local": "file:///data"}, "datasets": {}}), encoding="utf-8")

        with pytest.raises(ValueError, match="already registered"):
            registry.register_source("local", "file:///elsewhere")


class TestRegisterDataset:
    def test_requires_a_known_source_even_when_overriding(self, registry: DatasetRegistry) -> None:
        for override in (False, True):
            with pytest.raises(UnknownNameError, match="Source 'ghost' not registered"):
                registry.register_dataset("Toy", "ghost", "toy.h5mu", override=override)

    def test_accepts_a_builtin_source_and_persists(self, registry: DatasetRegistry, config_path: UPath) -> None:
        registry.register_dataset("Toy", _BUILTIN_SOURCE, "toy.h5mu")

        assert registry.datasets["Toy"] == DatasetEntry(source=_BUILTIN_SOURCE, file="toy.h5mu")
        assert _saved(config_path, "datasets") == {"Toy": {"source": _BUILTIN_SOURCE, "file": "toy.h5mu"}}

    def test_identical_repeat_leaves_the_file_untouched(self, registry: DatasetRegistry, config_path: UPath) -> None:
        registry.register_dataset("Toy", _BUILTIN_SOURCE, "toy.h5mu")
        config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        written = config_path.read_text(encoding="utf-8")

        registry.register_dataset("Toy", _BUILTIN_SOURCE, "toy.h5mu")

        assert config_path.read_text(encoding="utf-8") == written

    def test_identical_to_a_builtin_is_a_noop(self, registry: DatasetRegistry, config_path: UPath) -> None:
        builtin = registry.builtin_datasets[_BUILTIN_DATASET]

        registry.register_dataset(_BUILTIN_DATASET, builtin.source, builtin.file)

        assert registry.custom_datasets == {}
        assert not config_path.exists()

    def test_conflict_raises_and_keeps_the_original(self, registry: DatasetRegistry, config_path: UPath) -> None:
        registry.register_dataset("Toy", _BUILTIN_SOURCE, "toy.h5mu")

        with pytest.raises(ValueError, match="Dataset 'Toy' already registered"):
            registry.register_dataset("Toy", _BUILTIN_SOURCE, "other.h5mu")
        with pytest.raises(ValueError, match=f"Dataset '{_BUILTIN_DATASET}' already registered"):
            registry.register_dataset(_BUILTIN_DATASET, _BUILTIN_SOURCE, "override.h5mu")

        assert registry.custom_datasets["Toy"].file == "toy.h5mu"
        assert _BUILTIN_DATASET not in registry.custom_datasets

    def test_override_replaces_and_persists(self, registry: DatasetRegistry, config_path: UPath) -> None:
        registry.register_dataset("Toy", _BUILTIN_SOURCE, "toy.h5mu")

        registry.register_dataset("Toy", _BUILTIN_SOURCE, "other.h5mu", override=True)

        assert _saved(config_path, "datasets") == {"Toy": {"source": _BUILTIN_SOURCE, "file": "other.h5mu"}}


class TestUnregister:
    def test_removes_custom_entries(self, registry: DatasetRegistry) -> None:
        registry.register_source("local", "file:///data")
        registry.register_dataset("Toy", "local", "toy.h5mu")

        with pytest.raises(ValueError, match=r"still referenced by datasets \['Toy'\]"):
            registry.unregister_source("local")
        registry.unregister_dataset("Toy")
        registry.unregister_source("local")

        assert (registry.custom_datasets, registry.custom_sources) == ({}, {})

    def test_rejects_unknown_names(self, registry: DatasetRegistry) -> None:
        with pytest.raises(UnknownNameError, match="Dataset 'Toy' not in custom registry"):
            registry.unregister_dataset("Toy")
        with pytest.raises(UnknownNameError, match="Source 'local' not in custom registry"):
            registry.unregister_source("local")

    def test_refuses_builtins_even_when_overridden(self, registry: DatasetRegistry) -> None:
        registry.register_dataset(_BUILTIN_DATASET, _BUILTIN_SOURCE, "override.h5mu", override=True)
        registry.register_source(_BUILTIN_SOURCE, "file:///override", override=True)

        with pytest.raises(UnknownNameError, match="is built-in and cannot be unregistered"):
            registry.unregister_dataset(_BUILTIN_DATASET)
        with pytest.raises(UnknownNameError, match="is built-in and cannot be unregistered"):
            registry.unregister_source(_BUILTIN_SOURCE)


def test_reload_picks_up_external_edits(registry: DatasetRegistry, config_path: UPath) -> None:
    config_path.write_text(
        json.dumps({"sources": {}, "datasets": {"External": {"source": _BUILTIN_SOURCE, "file": "e.h5mu"}}}),
        encoding="utf-8",
    )

    registry.reload()

    assert "External" in registry.custom_datasets


def test_to_dataframe_labels_origins_and_repr_has_no_index(registry: DatasetRegistry) -> None:
    registry.register_dataset("Toy", _BUILTIN_SOURCE, "toy.h5mu")

    frame = registry.to_dataframe().set_index("Name")

    assert list(registry.to_dataframe().columns) == ["Name", "Source", "File", "Origin"]
    assert frame.loc["Toy", "Origin"] == "custom"
    assert frame.loc[_BUILTIN_DATASET, "Origin"] == "built-in"
    assert not repr(registry).startswith("0")


class TestFacade:
    def test_module_functions_delegate_to_the_singleton(self) -> None:
        assert _BUILTIN_DATASET in dataset_registry.builtin_datasets
        assert dataset_facade.list() == dataset_registry.dataset_names
        assert list(dataset_facade.table().columns) == ["Name", "Source", "File", "Origin"]

    def test_register_forwards_override(self, registry: DatasetRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dataset_facade, "dataset_registry", registry)
        dataset_facade.register_source("local", "file:///data")
        dataset_facade.register_dataset("Toy", _BUILTIN_SOURCE, "toy.h5mu")

        with pytest.raises(ValueError, match="already registered"):
            dataset_facade.register_source("local", "file:///elsewhere")
        with pytest.raises(ValueError, match="already registered"):
            dataset_facade.register_dataset("Toy", _BUILTIN_SOURCE, "other.h5mu")
        dataset_facade.register_source("local", "file:///elsewhere", override=True)
        dataset_facade.register_dataset("Toy", _BUILTIN_SOURCE, "other.h5mu", override=True)

        assert registry.custom_sources["local"].url == "file:///elsewhere"
        assert registry.custom_datasets["Toy"].file == "other.h5mu"
