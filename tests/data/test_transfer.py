"""Tests for streaming downloads through ``drevalpy.data._transfer``; the "remote" is a local directory."""

from __future__ import annotations

import os

import pytest
from upath import UPath

from drevalpy.data._transfer import _CHUNK_SIZE, _progress, _stream, download_file, download_files


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
