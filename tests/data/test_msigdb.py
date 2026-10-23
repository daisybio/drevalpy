"""Tests for the MSigDB gene-set download and cache used by Precily."""

from __future__ import annotations

import hashlib
from pathlib import Path

import aiohttp
import pytest

from drevalpy.data import _msigdb
from drevalpy.data._msigdb import GMT_FILENAME, GMT_SHA256, GMT_URL, get_precily_gmt
from drevalpy.errors import InputNotFoundError, InvalidInputError

FIXTURE_GMT = "SET_A\tsynthetic\tG1\tG2\tG3\tG4\tG5\nSET_B\tsynthetic\tG3\tG4\tG5\tG6\tG7\n"


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the cache at a temporary directory and pin the hash of the fixture GMT."""
    monkeypatch.setenv("DREVALPY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(_msigdb, "GMT_SHA256", hashlib.sha256(FIXTURE_GMT.encode()).hexdigest())
    return tmp_path / "cache" / "msigdb" / GMT_FILENAME


def _fake_download(content: str, calls: list[str]):
    def download(remote, local_path, label):
        calls.append(str(remote))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content, encoding="utf-8")
        return local_path

    return download


def test_pinned_source_is_msigdb_release_6_1() -> None:
    assert GMT_URL == "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/6.1/c2.cp.v6.1.symbols.gmt"
    assert len(GMT_SHA256) == 64


def test_first_use_downloads_into_the_cache(monkeypatch: pytest.MonkeyPatch, cache: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(_msigdb, "download_file", _fake_download(FIXTURE_GMT, calls))

    path = get_precily_gmt()

    assert Path(str(path)) == cache
    assert path.read_text(encoding="utf-8") == FIXTURE_GMT
    assert calls == [GMT_URL]


def test_cached_file_is_reused_without_downloading(monkeypatch: pytest.MonkeyPatch, cache: Path) -> None:
    cache.parent.mkdir(parents=True)
    cache.write_text(FIXTURE_GMT, encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(_msigdb, "download_file", _fake_download("unused", calls))

    assert Path(str(get_precily_gmt())) == cache
    assert calls == []


def test_cached_file_with_the_wrong_hash_is_replaced(monkeypatch: pytest.MonkeyPatch, cache: Path) -> None:
    cache.parent.mkdir(parents=True)
    cache.write_text("truncated", encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(_msigdb, "download_file", _fake_download(FIXTURE_GMT, calls))

    assert get_precily_gmt().read_text(encoding="utf-8") == FIXTURE_GMT
    assert calls == [GMT_URL]


def test_download_with_the_wrong_hash_is_deleted_and_reported(monkeypatch: pytest.MonkeyPatch, cache: Path) -> None:
    monkeypatch.setattr(_msigdb, "download_file", _fake_download("not the gene sets", []))

    with pytest.raises(InvalidInputError, match="do not match the pinned sha256") as info:
        get_precily_gmt()

    assert GMT_URL in str(info.value)
    assert not cache.exists()


@pytest.mark.usefixtures("cache")
@pytest.mark.parametrize("failure", [OSError("no route to host"), aiohttp.ClientConnectionError("refused")])
def test_download_failure_names_the_url(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    def failing(remote, local_path, label):
        raise failure

    monkeypatch.setattr(_msigdb, "download_file", failing)

    with pytest.raises(InputNotFoundError) as info:
        get_precily_gmt()

    message = str(info.value)
    assert GMT_URL in message
    assert str(failure) in message
    assert 'uns["pathways_gmt"]' in message
    assert message.startswith("Could not download the MSigDB gene sets")


@pytest.mark.network
def test_real_download_matches_the_pinned_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DREVALPY_CACHE_DIR", str(tmp_path / "real"))

    path = get_precily_gmt()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert hashlib.sha256(Path(str(path)).read_bytes()).hexdigest() == GMT_SHA256
    assert len(lines) == 1329
    assert get_precily_gmt() == path
