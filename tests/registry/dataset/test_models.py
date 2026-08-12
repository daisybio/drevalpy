"""Tests for the dataset-registry pydantic config models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drevalpy.registry.dataset._models import DatasetEntry, DrevalConfig, SourceEntry


def test_source_entry_defaults_to_no_storage_options() -> None:
    assert SourceEntry(url="s3://bucket/").storage_options == {}


def test_source_entry_from_raw_accepts_a_bare_url() -> None:
    assert SourceEntry.from_raw("s3://bucket/") == SourceEntry(url="s3://bucket/")


def test_source_entry_from_raw_accepts_a_mapping() -> None:
    entry = SourceEntry.from_raw({"url": "s3://bucket/", "storage_options": {"anon": True}})

    assert entry == SourceEntry(url="s3://bucket/", storage_options={"anon": True})


def test_source_entry_to_raw_collapses_to_a_string_without_options() -> None:
    assert SourceEntry(url="s3://bucket/").to_raw() == "s3://bucket/"


def test_source_entry_to_raw_keeps_a_mapping_with_options() -> None:
    entry = SourceEntry(url="s3://bucket/", storage_options={"anon": True})

    assert entry.to_raw() == {"url": "s3://bucket/", "storage_options": {"anon": True}}


def test_source_entry_requires_a_url() -> None:
    with pytest.raises(ValidationError, match="url"):
        SourceEntry()  # type: ignore[call-arg]


def test_dataset_entry_requires_source_and_file() -> None:
    with pytest.raises(ValidationError, match="file"):
        DatasetEntry(source="local")  # type: ignore[call-arg]


def test_config_defaults_are_empty() -> None:
    config = DrevalConfig()

    assert config.sources == {}
    assert config.datasets == {}


def test_config_rejects_unknown_root_keys() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DrevalConfig(cache_dir="/var/cache")  # type: ignore[call-arg]


def test_config_from_raw_ignores_absent_sections() -> None:
    assert DrevalConfig.from_raw({}) == DrevalConfig()


def test_config_from_raw_parses_mixed_source_shapes() -> None:
    config = DrevalConfig.from_raw(
        {
            "sources": {
                "plain": "https://example.org/",
                "with_options": {"url": "s3://bucket/", "storage_options": {"anon": True}},
            },
            "datasets": {"Toy": {"source": "plain", "file": "toy.h5mu"}},
        }
    )

    assert config.sources["plain"] == SourceEntry(url="https://example.org/")
    assert config.sources["with_options"].storage_options == {"anon": True}
    assert config.datasets["Toy"] == DatasetEntry(source="plain", file="toy.h5mu")


def test_config_to_raw_serializes_sources_and_datasets() -> None:
    config = DrevalConfig(
        sources={"plain": SourceEntry(url="https://example.org/")},
        datasets={"Toy": DatasetEntry(source="plain", file="toy.h5mu")},
    )

    assert config.to_raw() == {
        "sources": {"plain": "https://example.org/"},
        "datasets": {"Toy": {"source": "plain", "file": "toy.h5mu"}},
    }


def test_config_round_trips_through_raw() -> None:
    config = DrevalConfig(
        sources={"s3": SourceEntry(url="s3://bucket/", storage_options={"anon": True})},
        datasets={"Toy": DatasetEntry(source="s3", file="toy.h5mu")},
    )

    assert DrevalConfig.from_raw(config.to_raw()) == config
