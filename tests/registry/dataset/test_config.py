"""Tests for the dataset registry models and its on-disk config I/O."""

from __future__ import annotations

import json

import pytest
from filelock import FileLock, Timeout
from pydantic import ValidationError
from upath import UPath

from drevalpy.registry.dataset import _io
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


@pytest.fixture
def config_dir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> UPath:
    monkeypatch.setenv("DREVALPY_CONFIG_DIR", str(tmp_path))
    return UPath(tmp_path)


def test_config_and_lock_paths_live_in_the_config_dir(config_dir: UPath) -> None:
    assert _io.get_config_path() == config_dir / "datasets.json"
    assert _io._lock_path() == config_dir / "datasets.lock"


class TestConfigLock:
    def test_creates_the_lock_file_and_releases_on_exit(self, config_dir: UPath) -> None:
        with _io.config_lock():
            assert _io._lock_path().is_file()
        with _io.config_lock():
            assert _io._lock_path().is_file()

    def test_is_exclusive_while_held(self, config_dir: UPath) -> None:
        contender = FileLock(_io._lock_path(), timeout=0)

        with _io.config_lock(), pytest.raises(Timeout):
            contender.acquire()

    def test_is_reentrant_and_stays_held_until_the_outermost_block_exits(self, config_dir: UPath) -> None:
        contender = FileLock(_io._lock_path(), timeout=0)

        with _io.config_lock():
            with _io.config_lock():
                pass
            with pytest.raises(Timeout):
                contender.acquire()


def test_load_config_returns_defaults_when_the_file_is_missing(config_dir: UPath) -> None:
    assert _io.load_config() == DrevalConfig()


def test_load_config_parses_an_existing_file(config_dir: UPath) -> None:
    _io.get_config_path().write_text(
        json.dumps(
            {"sources": {"local": "file:///data"}, "datasets": {"Toy": {"source": "local", "file": "toy.h5mu"}}}
        ),
        encoding="utf-8",
    )

    config = _io.load_config()

    assert config.sources == {"local": SourceEntry(url="file:///data")}
    assert config.datasets == {"Toy": DatasetEntry(source="local", file="toy.h5mu")}


def test_save_config_creates_the_directory_and_writes_indented_json(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DREVALPY_CONFIG_DIR", str(tmp_path / "nested" / "config"))

    _io.save_config(DrevalConfig(sources={"local": SourceEntry(url="file:///data")}))
    written = _io.get_config_path().read_text(encoding="utf-8")

    assert written.endswith("\n")
    assert '\n  "sources"' in written


def test_save_then_load_round_trips_storage_options(config_dir: UPath) -> None:
    original = DrevalConfig(
        sources={"s3": SourceEntry(url="s3://bucket/", storage_options={"anon": True})},
        datasets={"Toy": DatasetEntry(source="s3", file="toy.h5mu")},
    )

    _io.save_config(original)

    assert _io.load_config() == original
