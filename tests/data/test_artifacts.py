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
    """Point cache and artifact env vars at temporary locations."""
    monkeypatch.setenv("DREVALPY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("DREVALPY_ARTIFACTS_URI", raising=False)
    monkeypatch.delenv("DREVALPY_ARTIFACTS_STORAGE_OPTIONS", raising=False)


class TestArtifactsUri:
    """Resolution of the artifacts base URI."""

    def test_defaults_to_bundled_bucket(self):
        assert get_artifacts_uri() == _DEFAULT_ARTIFACTS_URI

    def test_env_var_overrides(self, monkeypatch):
        monkeypatch.setenv("DREVALPY_ARTIFACTS_URI", "s3://mirror/artifacts/")
        assert get_artifacts_uri() == "s3://mirror/artifacts/"

    def test_blank_env_var_falls_back(self, monkeypatch):
        monkeypatch.setenv("DREVALPY_ARTIFACTS_URI", "   ")
        assert get_artifacts_uri() == _DEFAULT_ARTIFACTS_URI


class TestArtifactsStorageOptions:
    """Resolution of fsspec storage options."""

    def test_empty_by_default(self):
        """No hardcoded credentials, so the ambient credential chain applies."""
        assert get_artifacts_storage_options() == {}

    def test_parses_json_object(self, monkeypatch):
        monkeypatch.setenv("DREVALPY_ARTIFACTS_STORAGE_OPTIONS", '{"profile": "dev", "anon": false}')
        assert get_artifacts_storage_options() == {"profile": "dev", "anon": False}

    def test_invalid_json_is_ignored(self, monkeypatch):
        monkeypatch.setenv("DREVALPY_ARTIFACTS_STORAGE_OPTIONS", "{nope")
        assert get_artifacts_storage_options() == {}

    def test_non_object_json_is_ignored(self, monkeypatch):
        monkeypatch.setenv("DREVALPY_ARTIFACTS_STORAGE_OPTIONS", '["a", "b"]')
        assert get_artifacts_storage_options() == {}


class TestGetArtifact:
    """Downloading and caching of single-file artifacts."""

    @pytest.fixture
    def remote(self, tmp_path, monkeypatch):
        """Serve artifacts from a local directory instead of S3."""
        remote_dir = tmp_path / "remote"
        remote_dir.mkdir()
        monkeypatch.setenv("DREVALPY_ARTIFACTS_URI", f"{remote_dir}/")
        return remote_dir

    def test_downloads_file(self, remote):
        (remote / "weights.bin").write_bytes(b"payload")
        local = get_artifact("weights.bin")
        assert local.read_bytes() == b"payload"
        assert local.name == "weights.bin"

    def test_reuses_cached_file(self, remote):
        (remote / "weights.bin").write_bytes(b"payload")
        first = get_artifact("weights.bin")
        (remote / "weights.bin").unlink()
        assert get_artifact("weights.bin").read_bytes() == first.read_bytes()

    def test_leaves_no_partial_files(self, remote):
        (remote / "weights.bin").write_bytes(b"payload")
        local = get_artifact("weights.bin")
        assert not list(UPath(local.parent).glob("*.part"))

    def test_missing_artifact_raises(self, remote):
        with pytest.raises(FileNotFoundError):
            get_artifact("absent.bin")


class TestGetArtifactDir:
    """Downloading and caching of multi-file artifacts."""

    @pytest.fixture
    def remote(self, tmp_path, monkeypatch):
        """Serve a two-file artifact directory from a local directory."""
        remote_dir = tmp_path / "remote" / "model"
        remote_dir.mkdir(parents=True)
        (remote_dir / "config.json").write_text("{}")
        (remote_dir / "weights.bin").write_bytes(b"payload")
        monkeypatch.setenv("DREVALPY_ARTIFACTS_URI", f"{remote_dir.parent}/")
        return remote_dir

    def test_downloads_all_files(self, remote):
        local = get_artifact_dir("model", ("config.json", "weights.bin"))
        assert (local / "config.json").read_text() == "{}"
        assert (local / "weights.bin").read_bytes() == b"payload"

    def test_reuses_complete_cache(self, remote):
        get_artifact_dir("model", ("config.json", "weights.bin"))
        (remote / "weights.bin").unlink()
        local = get_artifact_dir("model", ("config.json", "weights.bin"))
        assert (local / "weights.bin").read_bytes() == b"payload"

    def test_incomplete_cache_is_refetched(self, remote):
        local = get_artifact_dir("model", ("config.json", "weights.bin"))
        (local / "weights.bin").unlink()
        refetched = get_artifact_dir("model", ("config.json", "weights.bin"))
        assert (refetched / "weights.bin").read_bytes() == b"payload"

    def test_only_requested_files_are_fetched(self, remote):
        local = get_artifact_dir("model", ("config.json",))
        assert (local / "config.json").exists()
        assert not (local / "weights.bin").exists()
