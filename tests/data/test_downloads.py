"""Tests for download and cache plumbing: cache directory, artifacts, streamed transfers and the MSigDB gene sets."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import aiohttp
import pytest
from platformdirs import user_cache_dir
from upath import UPath

from drevalpy.data import _msigdb
from drevalpy.data._msigdb import GMT_FILENAME, GMT_SHA256, GMT_URL, get_precily_gmt
from drevalpy.data._paths import get_default_data_dir
from drevalpy.data._transfer import _CHUNK_SIZE, _progress, _stream, download_file, download_files
from drevalpy.data.artifacts import (
    _DEFAULT_ARTIFACTS_URI,
    get_artifact,
    get_artifact_dir,
    get_artifacts_storage_options,
    get_artifacts_uri,
)
from drevalpy.errors import InputNotFoundError, InvalidInputError


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DREVALPY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("DREVALPY_ARTIFACTS_URI", raising=False)
    monkeypatch.delenv("DREVALPY_ARTIFACTS_STORAGE_OPTIONS", raising=False)


def test_artifacts_uri_defaults_to_the_bundled_bucket_and_env_overrides_unless_blank(monkeypatch):
    assert get_artifacts_uri() == _DEFAULT_ARTIFACTS_URI
    monkeypatch.setenv("DREVALPY_ARTIFACTS_URI", "s3://mirror/artifacts/")
    assert get_artifacts_uri() == "s3://mirror/artifacts/"
    monkeypatch.setenv("DREVALPY_ARTIFACTS_URI", "   ")
    assert get_artifacts_uri() == _DEFAULT_ARTIFACTS_URI


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, {}),
        ('{"profile": "dev", "anon": false}', {"profile": "dev", "anon": False}),
        ("{nope", {}),
        ('["a", "b"]', {}),
    ],
    ids=["unset-uses-ambient-credentials", "json-object", "invalid-json-ignored", "non-object-ignored"],
)
def test_storage_options_are_parsed_from_a_json_object(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv("DREVALPY_ARTIFACTS_STORAGE_OPTIONS", value)

    assert get_artifacts_storage_options() == expected


class TestGetArtifact:
    @pytest.fixture
    def remote(self, tmp_path, monkeypatch):
        remote_dir = tmp_path / "remote"
        remote_dir.mkdir()
        monkeypatch.setenv("DREVALPY_ARTIFACTS_URI", f"{remote_dir}/")
        return remote_dir

    def test_downloads_once_leaving_no_partial_files_and_reuses_the_cache(self, remote):
        (remote / "weights.bin").write_bytes(b"payload")

        local = get_artifact("weights.bin")
        (remote / "weights.bin").unlink()

        assert local.read_bytes() == b"payload"
        assert local.name == "weights.bin"
        assert not list(UPath(local.parent).glob("*.part"))
        assert get_artifact("weights.bin").read_bytes() == b"payload"

    def test_missing_artifact_raises(self, remote):
        with pytest.raises(FileNotFoundError):
            get_artifact("absent.bin")


class TestGetArtifactDir:
    @pytest.fixture
    def remote(self, tmp_path, monkeypatch):
        remote_dir = tmp_path / "remote" / "model"
        remote_dir.mkdir(parents=True)
        (remote_dir / "config.json").write_text("{}")
        (remote_dir / "weights.bin").write_bytes(b"payload")
        monkeypatch.setenv("DREVALPY_ARTIFACTS_URI", f"{remote_dir.parent}/")
        return remote_dir

    def test_downloads_all_files_reuses_a_complete_cache_and_refetches_an_incomplete_one(self, remote):
        files = ("config.json", "weights.bin")
        local = get_artifact_dir("model", files)
        assert (local / "config.json").read_text() == "{}"
        assert (local / "weights.bin").read_bytes() == b"payload"

        (remote / "weights.bin").unlink()
        assert (get_artifact_dir("model", files) / "weights.bin").read_bytes() == b"payload"

        (local / "weights.bin").unlink()
        (remote / "weights.bin").write_bytes(b"payload")
        assert (get_artifact_dir("model", files) / "weights.bin").read_bytes() == b"payload"

    def test_only_requested_files_are_fetched(self, remote):
        local = get_artifact_dir("model", ("config.json",))

        assert (local / "config.json").exists()
        assert not (local / "weights.bin").exists()


@pytest.fixture
def remote(tmp_path) -> UPath:
    remote_dir = UPath(tmp_path) / "remote"
    remote_dir.mkdir()
    return remote_dir


@pytest.fixture
def local(tmp_path) -> UPath:
    """Destination directory, deliberately not created up front."""
    return UPath(tmp_path) / "local"


class TestDownloadFile:
    def test_copies_the_payload_to_the_destination_creating_parents_without_partial_files(
        self, remote: UPath, local: UPath
    ) -> None:
        (remote / "weights.bin").write_bytes(b"payload")

        result = download_file(remote / "weights.bin", local / "nested" / "weights.bin", "weights")

        assert result == local / "nested" / "weights.bin"
        assert result.read_bytes() == b"payload"
        assert not list(local.rglob("*.part"))

    def test_copies_payloads_larger_than_one_chunk(self, remote: UPath, local: UPath) -> None:
        payload = os.urandom(_CHUNK_SIZE + 17)
        (remote / "weights.bin").write_bytes(payload)

        assert download_file(remote / "weights.bin", local / "weights.bin", "weights").read_bytes() == payload

    def test_missing_source_raises_and_leaves_no_partial_file(self, remote: UPath, local: UPath) -> None:
        with pytest.raises(FileNotFoundError):
            download_file(remote / "absent.bin", local / "absent.bin", "absent")

        assert not list(local.glob("*.part"))


class TestDownloadFiles:
    def test_copies_only_the_requested_files(self, remote: UPath, local: UPath) -> None:
        (remote / "config.json").write_text("{}")
        (remote / "weights.bin").write_bytes(b"payload")

        result = download_files(remote, local, "model", ("config.json",))

        assert (result / "config.json").read_text() == "{}"
        assert not (local / "weights.bin").exists()
        assert not list(local.glob("*.part"))

    def test_an_empty_filename_list_still_creates_the_directory(self, remote: UPath, local: UPath) -> None:
        result = download_files(remote, local, "model", ())

        assert result.is_dir()
        assert list(result.iterdir()) == []

    def test_a_missing_member_leaves_no_partial_file_behind(self, remote: UPath, local: UPath) -> None:
        (remote / "config.json").write_text("{}")

        with pytest.raises(FileNotFoundError):
            download_files(remote, local, "model", ("config.json", "absent.bin"))

        assert not list(local.glob("*.part"))


class TestStream:
    def test_stages_through_a_part_file_and_overwrites_an_existing_destination(
        self, remote: UPath, local: UPath
    ) -> None:
        (remote / "weights.bin").write_bytes(b"new")
        local.mkdir(parents=True)
        (local / "weights.bin").write_bytes(b"stale")

        with _progress() as progress:
            _stream(remote / "weights.bin", local / "weights.bin", "weights", progress)

        assert [path.name for path in local.iterdir()] == ["weights.bin"]
        assert (local / "weights.bin").read_bytes() == b"new"

    def test_destination_only_appears_once_complete(self, remote: UPath, local: UPath, monkeypatch) -> None:
        """``os.replace`` publishes the file, so a failed write leaves nothing."""
        (remote / "weights.bin").write_bytes(b"payload")
        local.mkdir(parents=True)

        def _failing_replace(src: object, dst: object) -> None:
            raise OSError("cross-device link")

        monkeypatch.setattr("drevalpy.data._transfer.os.replace", _failing_replace)

        with _progress() as progress, pytest.raises(OSError, match="cross-device link"):
            _stream(remote / "weights.bin", local / "weights.bin", "weights", progress)

        assert not (local / "weights.bin").exists()
        assert not list(local.glob("*.part"))


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
        local_path.write_bytes(content.encode())
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
    cache.write_bytes(FIXTURE_GMT.encode())
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


@pytest.mark.parametrize("pad", ["", "  "])
def test_env_var_override_is_stripped(monkeypatch, tmp_path, pad):
    custom = str(tmp_path / "custom_cache")
    monkeypatch.setenv("DREVALPY_CACHE_DIR", f"{pad}{custom}{pad}")
    assert get_default_data_dir() == Path(custom)


@pytest.mark.parametrize("value", ["", None])
def test_empty_or_unset_env_var_uses_platformdirs(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("DREVALPY_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("DREVALPY_CACHE_DIR", value)
    assert get_default_data_dir() == Path(user_cache_dir("drevalpy"))
