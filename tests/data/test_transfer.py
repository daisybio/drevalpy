"""Tests for streaming downloads through ``drevalpy.data._transfer``.

The "remote" is a plain ``tmp_path`` directory: ``UPath`` resolves it through the
local fsspec filesystem, so the same ``fs.open`` / ``fs.size`` code path runs as
for S3 without any network access.
"""

from __future__ import annotations

import os

import pytest
from upath import UPath

from drevalpy.data._transfer import _CHUNK_SIZE, _progress, _stream, download_file, download_files


@pytest.fixture
def remote(tmp_path) -> UPath:
    """Serve files from a local directory instead of S3."""
    remote_dir = UPath(tmp_path) / "remote"
    remote_dir.mkdir()
    return remote_dir


@pytest.fixture
def local(tmp_path) -> UPath:
    """Destination directory, deliberately not created up front."""
    return UPath(tmp_path) / "local"


class TestDownloadFile:
    """Single-file downloads."""

    def test_copies_the_payload(self, remote: UPath, local: UPath) -> None:
        (remote / "weights.bin").write_bytes(b"payload")

        result = download_file(remote / "weights.bin", local / "weights.bin", "weights")

        assert result.read_bytes() == b"payload"

    def test_returns_the_destination_path(self, remote: UPath, local: UPath) -> None:
        (remote / "weights.bin").write_bytes(b"payload")

        result = download_file(remote / "weights.bin", local / "weights.bin", "weights")

        assert result == local / "weights.bin"

    def test_creates_missing_parent_directories(self, remote: UPath, local: UPath) -> None:
        (remote / "weights.bin").write_bytes(b"payload")

        download_file(remote / "weights.bin", local / "nested" / "weights.bin", "weights")

        assert (local / "nested").is_dir()

    def test_leaves_no_partial_files(self, remote: UPath, local: UPath) -> None:
        (remote / "weights.bin").write_bytes(b"payload")

        download_file(remote / "weights.bin", local / "weights.bin", "weights")

        assert not list(local.glob("*.part"))

    def test_copies_payloads_larger_than_one_chunk(self, remote: UPath, local: UPath) -> None:
        payload = os.urandom(_CHUNK_SIZE + 17)
        (remote / "weights.bin").write_bytes(payload)

        result = download_file(remote / "weights.bin", local / "weights.bin", "weights")

        assert result.read_bytes() == payload

    def test_missing_source_raises(self, remote: UPath, local: UPath) -> None:
        with pytest.raises(FileNotFoundError):
            download_file(remote / "absent.bin", local / "absent.bin", "absent")

    def test_missing_source_leaves_no_partial_file(self, remote: UPath, local: UPath) -> None:
        with pytest.raises(FileNotFoundError):
            download_file(remote / "absent.bin", local / "absent.bin", "absent")

        assert not list(local.glob("*.part"))


class TestDownloadFiles:
    """Multi-file downloads into a shared directory."""

    def test_copies_every_requested_file(self, remote: UPath, local: UPath) -> None:
        (remote / "config.json").write_text("{}")
        (remote / "weights.bin").write_bytes(b"payload")

        result = download_files(remote, local, "model", ("config.json", "weights.bin"))

        assert (result / "config.json").read_text() == "{}"
        assert (result / "weights.bin").read_bytes() == b"payload"

    def test_skips_files_that_were_not_requested(self, remote: UPath, local: UPath) -> None:
        (remote / "config.json").write_text("{}")
        (remote / "weights.bin").write_bytes(b"payload")

        download_files(remote, local, "model", ("config.json",))

        assert not (local / "weights.bin").exists()

    def test_creates_the_destination_directory(self, remote: UPath, local: UPath) -> None:
        (remote / "config.json").write_text("{}")

        result = download_files(remote, local, "model", ("config.json",))

        assert result.is_dir()

    def test_empty_filename_list_still_creates_the_directory(self, remote: UPath, local: UPath) -> None:
        result = download_files(remote, local, "model", ())

        assert result.is_dir()
        assert list(result.iterdir()) == []

    def test_leaves_no_partial_files(self, remote: UPath, local: UPath) -> None:
        (remote / "config.json").write_text("{}")
        (remote / "weights.bin").write_bytes(b"payload")

        download_files(remote, local, "model", ("config.json", "weights.bin"))

        assert not list(local.glob("*.part"))

    def test_a_missing_member_leaves_no_partial_file_behind(self, remote: UPath, local: UPath) -> None:
        (remote / "config.json").write_text("{}")

        with pytest.raises(FileNotFoundError):
            download_files(remote, local, "model", ("config.json", "absent.bin"))

        assert not list(local.glob("*.part"))


class TestStream:
    """Staging behaviour of the private copy helper."""

    def test_stages_through_a_pid_scoped_part_file(self, remote: UPath, local: UPath) -> None:
        (remote / "weights.bin").write_bytes(b"payload")
        local.mkdir(parents=True)
        observed: list[str] = []

        with _progress() as progress:
            _stream(remote / "weights.bin", local / "weights.bin", "weights", progress)
            observed.extend(path.name for path in local.iterdir())

        assert observed == ["weights.bin"]

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

    def test_overwrites_an_existing_destination(self, remote: UPath, local: UPath) -> None:
        (remote / "weights.bin").write_bytes(b"new")
        local.mkdir(parents=True)
        (local / "weights.bin").write_bytes(b"stale")

        with _progress() as progress:
            _stream(remote / "weights.bin", local / "weights.bin", "weights", progress)

        assert (local / "weights.bin").read_bytes() == b"new"
