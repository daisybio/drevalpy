"""Tests for the dataset-registry pydantic config models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drevalpy.registry.dataset._models import DatasetEntry, DrevalConfig, SourceEntry


def test_source_entry_raw_forms() -> None:
    options = {"anon": True}

    assert SourceEntry.from_raw("s3://bucket/") == SourceEntry(url="s3://bucket/")
    assert SourceEntry.from_raw({"url": "s3://bucket/", "storage_options": options}) == SourceEntry(
        url="s3://bucket/", storage_options=options
    )
    assert SourceEntry(url="s3://bucket/").to_raw() == "s3://bucket/"
    assert SourceEntry(url="s3://bucket/", storage_options=options).to_raw() == {
        "url": "s3://bucket/",
        "storage_options": options,
    }


@pytest.mark.parametrize(
    ("build", "field"),
    [
        (lambda: SourceEntry(), "url"),  # type: ignore[call-arg]
        (lambda: DatasetEntry(source="local"), "file"),  # type: ignore[call-arg]
    ],
    ids=["source-url", "dataset-file"],
)
def test_required_fields(build, field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        build()


def test_config_rejects_unknown_root_keys() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DrevalConfig(cache_dir="/var/cache")  # type: ignore[call-arg]


def test_config_from_raw_ignores_absent_sections_and_parses_mixed_source_shapes() -> None:
    assert DrevalConfig.from_raw({}) == DrevalConfig()

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


def test_config_round_trips_through_raw_including_an_optional_hash() -> None:
    digest = "ab" * 32
    config = DrevalConfig(
        sources={
            "plain": SourceEntry(url="https://example.org/"),
            "s3": SourceEntry(url="s3://bucket/", storage_options={"anon": True}),
        },
        datasets={
            "Plain": DatasetEntry(source="plain", file="toy.h5mu"),
            "Pinned": DatasetEntry(source="s3", file="toy.h5mu", sha256=digest),
        },
    )

    raw = config.to_raw()

    assert raw["sources"]["plain"] == "https://example.org/"
    assert raw["datasets"] == {
        "Plain": {"source": "plain", "file": "toy.h5mu"},
        "Pinned": {"source": "s3", "file": "toy.h5mu", "sha256": digest},
    }
    assert DrevalConfig.from_raw(raw) == config
