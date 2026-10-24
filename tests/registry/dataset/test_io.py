"""Tests for dataset-registry config file I/O; ``DREVALPY_CONFIG_DIR`` keeps them off the real config."""

from __future__ import annotations

import json

import pytest
from filelock import FileLock, Timeout
from upath import UPath

from drevalpy.registry.dataset import _io
from drevalpy.registry.dataset._models import DatasetEntry, DrevalConfig, SourceEntry


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
