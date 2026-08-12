"""Tests for :class:`~drevalpy.registry.dataset._registry.DatasetRegistry`.

Every test drives a freshly constructed ``DatasetRegistry`` rather than the
``dataset_registry`` singleton, and every registration is written into a
``tmp_path`` config directory, so neither the module singleton nor the
developer's real ``datasets.json`` is mutated.
"""

from __future__ import annotations

import json

import pytest
from upath import UPath

from drevalpy.registry import dataset as dataset_facade
from drevalpy.registry.dataset._models import DatasetEntry, SourceEntry
from drevalpy.registry.dataset._registry import DatasetRegistry, dataset_registry

_BUILTIN_DATASET = "GDSC1"
_BUILTIN_SOURCE = "orakl_s3"


@pytest.fixture
def config_path(tmp_path, monkeypatch: pytest.MonkeyPatch) -> UPath:
    """Redirect the drevalpy config directory into ``tmp_path``."""
    monkeypatch.setenv("DREVALPY_CONFIG_DIR", str(tmp_path))
    return UPath(tmp_path) / "datasets.json"


@pytest.fixture
def registry(config_path: UPath) -> DatasetRegistry:
    """A registry whose custom entries live in an empty ``tmp_path`` config."""
    return DatasetRegistry()


def test_builtin_datasets_are_loaded_from_the_packaged_json(registry: DatasetRegistry) -> None:
    assert registry.builtin_datasets[_BUILTIN_DATASET] == DatasetEntry(
        source=_BUILTIN_SOURCE, file=f"{_BUILTIN_DATASET}.h5mu"
    )


def test_builtin_sources_are_loaded_from_the_packaged_json(registry: DatasetRegistry) -> None:
    assert registry.builtin_sources[_BUILTIN_SOURCE].url.startswith("s3://")


def test_custom_entries_are_empty_without_a_config_file(registry: DatasetRegistry) -> None:
    assert registry.custom_datasets == {}
    assert registry.custom_sources == {}


def test_dataset_names_are_sorted(registry: DatasetRegistry) -> None:
    assert registry.dataset_names == sorted(registry.dataset_names)


def test_source_names_are_sorted(registry: DatasetRegistry) -> None:
    assert registry.source_names == sorted(registry.source_names)


def test_is_registered_recognizes_a_builtin(registry: DatasetRegistry) -> None:
    assert registry.is_registered(_BUILTIN_DATASET) is True


def test_is_registered_rejects_an_unknown_name(registry: DatasetRegistry) -> None:
    assert registry.is_registered("NotADataset") is False


def test_register_source_persists_the_entry(registry: DatasetRegistry, config_path: UPath) -> None:
    registry.register_source("local", "file:///data")

    assert json.loads(config_path.read_text(encoding="utf-8"))["sources"] == {"local": "file:///data"}


def test_register_source_keeps_storage_options(registry: DatasetRegistry) -> None:
    registry.register_source("bucket", "s3://bucket/", {"anon": True})

    assert registry.custom_sources["bucket"] == SourceEntry(url="s3://bucket/", storage_options={"anon": True})


def test_custom_sources_override_builtins(registry: DatasetRegistry) -> None:
    registry.register_source(_BUILTIN_SOURCE, "file:///override")

    assert registry.sources[_BUILTIN_SOURCE].url == "file:///override"
    assert registry.builtin_sources[_BUILTIN_SOURCE].url != "file:///override"


def test_register_dataset_requires_a_known_source(registry: DatasetRegistry) -> None:
    with pytest.raises(KeyError, match="Source 'ghost' not registered"):
        registry.register_dataset("Toy", "ghost", "toy.h5mu")


def test_register_dataset_accepts_a_builtin_source(registry: DatasetRegistry) -> None:
    registry.register_dataset("Toy", _BUILTIN_SOURCE, "toy.h5mu")

    assert registry.datasets["Toy"] == DatasetEntry(source=_BUILTIN_SOURCE, file="toy.h5mu")


def test_register_dataset_persists_the_entry(registry: DatasetRegistry, config_path: UPath) -> None:
    registry.register_source("local", "file:///data")
    registry.register_dataset("Toy", "local", "toy.h5mu")

    assert json.loads(config_path.read_text(encoding="utf-8"))["datasets"] == {
        "Toy": {"source": "local", "file": "toy.h5mu"}
    }


def test_unregister_dataset_removes_a_custom_entry(registry: DatasetRegistry) -> None:
    registry.register_source("local", "file:///data")
    registry.register_dataset("Toy", "local", "toy.h5mu")

    registry.unregister_dataset("Toy")

    assert registry.custom_datasets == {}


def test_unregister_dataset_rejects_an_unknown_name(registry: DatasetRegistry) -> None:
    with pytest.raises(KeyError, match="Dataset 'Toy' not in custom registry"):
        registry.unregister_dataset("Toy")


def test_unregister_dataset_refuses_a_builtin(registry: DatasetRegistry) -> None:
    registry.register_dataset(_BUILTIN_DATASET, _BUILTIN_SOURCE, "override.h5mu")

    with pytest.raises(KeyError, match="is built-in and cannot be unregistered"):
        registry.unregister_dataset(_BUILTIN_DATASET)


def test_unregister_source_removes_a_custom_entry(registry: DatasetRegistry) -> None:
    registry.register_source("local", "file:///data")

    registry.unregister_source("local")

    assert registry.custom_sources == {}


def test_unregister_source_rejects_an_unknown_name(registry: DatasetRegistry) -> None:
    with pytest.raises(KeyError, match="Source 'local' not in custom registry"):
        registry.unregister_source("local")


def test_unregister_source_refuses_a_builtin(registry: DatasetRegistry) -> None:
    registry.register_source(_BUILTIN_SOURCE, "file:///override")

    with pytest.raises(KeyError, match="is built-in and cannot be unregistered"):
        registry.unregister_source(_BUILTIN_SOURCE)


def test_unregister_source_refuses_a_referenced_source(registry: DatasetRegistry) -> None:
    registry.register_source("local", "file:///data")
    registry.register_dataset("Toy", "local", "toy.h5mu")

    with pytest.raises(ValueError, match=r"still referenced by datasets \['Toy'\]"):
        registry.unregister_source("local")


def test_reload_picks_up_external_edits(registry: DatasetRegistry, config_path: UPath) -> None:
    assert registry.custom_datasets == {}
    config_path.write_text(
        json.dumps({"sources": {}, "datasets": {"External": {"source": _BUILTIN_SOURCE, "file": "e.h5mu"}}}),
        encoding="utf-8",
    )

    registry.reload()

    assert "External" in registry.custom_datasets


def test_to_dataframe_labels_builtin_and_custom_origins(registry: DatasetRegistry) -> None:
    registry.register_dataset("Toy", _BUILTIN_SOURCE, "toy.h5mu")

    frame = registry.to_dataframe()

    assert list(frame.columns) == ["Name", "Source", "File", "Origin"]
    assert frame.set_index("Name").loc["Toy", "Origin"] == "custom"
    assert frame.set_index("Name").loc[_BUILTIN_DATASET, "Origin"] == "built-in"


def test_repr_renders_without_an_index(registry: DatasetRegistry) -> None:
    rendered = repr(registry)

    assert _BUILTIN_DATASET in rendered
    assert not rendered.startswith("0")


def test_repr_html_emits_a_table(registry: DatasetRegistry) -> None:
    assert "<table" in registry._repr_html_()


def test_module_singleton_exposes_the_builtin_datasets() -> None:
    assert _BUILTIN_DATASET in dataset_registry.builtin_datasets


def test_module_list_delegates_to_the_singleton() -> None:
    assert dataset_facade.list() == dataset_registry.dataset_names


def test_module_table_delegates_to_the_singleton() -> None:
    assert list(dataset_facade.table().columns) == ["Name", "Source", "File", "Origin"]
