"""Tests for artifact location resolution and local caching."""

from __future__ import annotations

import pytest
from upath import UPath

from drevalpy.data.artifacts import (
    _DEFAULT_ARTIFACTS_URI,
    get_artifact,
    get_artifact_dir,
    get_artifacts_storage_options,
    get_artifacts_uri,
)


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
