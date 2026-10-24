"""Tests for the dataset registry and loader."""

from __future__ import annotations

import hashlib
import json
import shutil
from importlib import resources
from pathlib import Path

import pytest
from upath import UPath

from drevalpy.data.datasets import _load as load_module
from drevalpy.data.datasets._load import _carries_curve_quality, _file_sha256, _load_from_cache, load
from drevalpy.errors import InputNotFoundError, InvalidInputError
from drevalpy.registry.dataset import DatasetRegistry as Registry
from drevalpy.registry.dataset import DrevalConfig, SourceEntry, get_config_path
from drevalpy.registry.dataset import dataset_registry as registry
from drevalpy.testing.synthetic import build_synthetic_dataset
from drevalpy.types.data.dataset import Dataset

_CORE_DATASETS = [
    "BeatAML2",
    "CTRPv1",
    "CTRPv2",
    "GDSC1",
    "GDSC2",
    "PDX_Bruna",
]


def _packaged_registry() -> dict:
    registry_path = resources.files("drevalpy.data.datasets").joinpath("available_datasets.json")
    with registry_path.open(encoding="utf-8") as handle:
        return json.load(handle)


class TestBuiltinRegistry:
    def test_available_datasets_json_structure(self) -> None:
        raw = _packaged_registry()

        assert "sources" in raw
        assert "datasets" in raw
        assert all(isinstance(val, (str, dict)) for val in raw["sources"].values())
        for entry in raw["datasets"].values():
            assert entry["source"] in raw["sources"]
            assert entry["file"].endswith(".h5mu")

    def test_names_sources_and_registration(self) -> None:
        assert registry.dataset_names == sorted(registry.dataset_names)
        assert set(_packaged_registry()["datasets"]) <= set(registry.dataset_names)
        assert set(_CORE_DATASETS) <= set(registry.datasets)
        assert "orakl" in registry.source_names
        assert registry.is_registered("GDSC1")
        assert not registry.is_registered("NonExistent")
        assert all(source.url for source in registry.sources.values())

    def test_builtin_vs_custom_separation(self) -> None:
        assert set(registry.builtin_datasets) == set(_packaged_registry()["datasets"])
        assert set(registry.builtin_sources) == set(_packaged_registry()["sources"])
        assert registry.custom_datasets.keys() <= registry.datasets.keys()

    def test_str_lists_datasets_and_sources_and_repr_equals_str(self) -> None:
        output = str(registry)

        assert all(name in output for name in ("GDSC1", "BeatAML2", "orakl"))
        assert str(registry) == repr(registry)

    def test_toyv1_is_pinned_to_the_gen4_artifact(self) -> None:
        entry = registry.datasets["TOYv1"]

        assert entry.source == "orakl_gen4"
        assert registry.sources["orakl_gen4"].url.endswith("/drevalpy_h5mu/gen4")
        assert entry.sha256 is not None
        assert len(entry.sha256) == 64


class TestRegistration:
    @pytest.fixture(autouse=True)
    def _use_tmp_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DREVALPY_CONFIG_DIR", str(tmp_path))
        self.reg = Registry()

    def test_register_and_unregister(self) -> None:
        self.reg.register_source("src", "https://example.com/data")
        self.reg.register_source("s3_src", "s3://bucket/path", storage_options={"profile": "dev"})
        self.reg.register_dataset("MyData", source="src", file="MyData.h5mu")

        assert self.reg.sources["src"].url == "https://example.com/data"
        assert self.reg.sources["s3_src"].storage_options == {"profile": "dev"}
        assert self.reg.datasets["MyData"].file == "MyData.h5mu"

        self.reg.unregister_dataset("MyData")
        self.reg.unregister_source("src")
        assert not self.reg.is_registered("MyData")
        assert "src" not in self.reg.custom_sources

    def test_registration_errors(self) -> None:
        self.reg.register_source("src", "https://example.com")
        self.reg.register_dataset("MyData", source="src", file="x.h5mu")

        with pytest.raises(KeyError, match="not registered"):
            self.reg.register_dataset("Other", source="nonexistent", file="x.h5mu")
        with pytest.raises(KeyError, match="not in custom"):
            self.reg.unregister_dataset("NonExistent")
        with pytest.raises(KeyError, match="not in custom"):
            self.reg.unregister_source("NonExistent")
        with pytest.raises(ValueError, match="still referenced"):
            self.reg.unregister_source("src")

    def test_custom_overrides_builtin(self) -> None:
        self.reg.register_source("orakl", "s3://my-mirror/data", override=True)

        assert self.reg.sources["orakl"].url == "s3://my-mirror/data"

    def test_registration_persists_to_disk_and_reload_picks_up_external_changes(self) -> None:
        self.reg.register_source("persisted", "https://example.com")
        self.reg.register_source("second", "https://second.com")
        config_path = get_config_path()
        assert "persisted" in json.loads(config_path.read_text())["sources"]
        assert {"persisted", "second"} <= set(self.reg.custom_sources)

        config_path.write_text(
            json.dumps(
                {
                    "sources": {"external": "https://external.org"},
                    "datasets": {"ExtData": {"source": "external", "file": "ext.h5mu"}},
                }
            )
        )
        self.reg.reload()

        assert self.reg.is_registered("ExtData")
        assert "external" in self.reg.custom_sources


class TestModels:
    def test_source_entry_round_trips(self) -> None:
        assert SourceEntry.from_raw("https://example.com").url == "https://example.com"
        assert SourceEntry.from_raw("https://example.com").storage_options == {}
        entry = SourceEntry.from_raw({"url": "s3://bucket", "storage_options": {"anon": True}})
        assert (entry.url, entry.storage_options) == ("s3://bucket", {"anon": True})
        assert SourceEntry(url="https://example.com").to_raw() == "https://example.com"
        assert SourceEntry(url="s3://bucket", storage_options={"profile": "dev"}).to_raw() == {
            "url": "s3://bucket",
            "storage_options": {"profile": "dev"},
        }

    def test_dreval_config_roundtrip_and_forbids_extra_keys(self) -> None:
        from pydantic import ValidationError

        raw = {
            "sources": {"lab": "https://lab.org/data"},
            "datasets": {"Study1": {"source": "lab", "file": "Study1.h5mu"}},
        }

        assert DrevalConfig.from_raw(raw).to_raw() == raw
        with pytest.raises(ValidationError):
            DrevalConfig(sources={}, datasets={}, unknown="bad")


def _stale_dataset():
    dataset = build_synthetic_dataset(n_samples=4, n_drugs=3)
    del dataset.response.layers["relevance_score"]
    return dataset


class TestCachedFileIsRejectedWhenTooOld:
    """A cache filled by an older drevalpy holds files without the quality layers the splitters read."""

    def test_the_quality_layers_decide_whether_a_dataset_is_usable(self) -> None:
        assert _carries_curve_quality(build_synthetic_dataset(n_samples=4, n_drugs=3))
        assert not _carries_curve_quality(_stale_dataset())

    def test_a_stale_cached_file_is_deleted_and_reported(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = UPath(tmp_path) / "stale.h5mu"
        _stale_dataset().save(path)

        with caplog.at_level("WARNING"):
            assert _load_from_cache(path) is None

        assert not path.is_file()
        assert "curve-quality" in caplog.text

    def test_a_usable_cached_file_is_returned_and_kept(self, tmp_path: Path) -> None:
        path = UPath(tmp_path) / "fresh.h5mu"
        build_synthetic_dataset(n_samples=4, n_drugs=3).save(path)

        assert _load_from_cache(path) is not None
        assert path.is_file()

    @pytest.mark.parametrize("kind", ["corrupt", "truncated"])
    def test_an_unreadable_cached_file_is_deleted(self, tmp_path: Path, kind: str) -> None:
        path = UPath(tmp_path) / f"{kind}.h5mu"
        if kind == "corrupt":
            path.write_bytes(b"not an h5mu file")
        else:
            build_synthetic_dataset(n_samples=4, n_drugs=3).save(path)
            path.write_bytes(path.read_bytes()[:200])

        assert _load_from_cache(path) is None
        assert not path.is_file()

    @pytest.mark.parametrize("error", [PermissionError("denied"), ImportError("missing optional dependency")])
    def test_a_permission_or_environment_error_keeps_the_file_and_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        path = UPath(tmp_path) / "locked.h5mu"
        path.write_bytes(b"x")

        def fail(_path: object) -> None:
            raise error

        monkeypatch.setattr(Dataset, "load", staticmethod(fail))

        with pytest.raises(type(error)):
            _load_from_cache(path)
        assert path.is_file()


def test_a_name_that_is_neither_cached_registered_nor_a_path_raises_input_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DREVALPY_CACHE_DIR", str(tmp_path))

    with pytest.raises(InputNotFoundError, match="Cannot locate .h5mu for dataset 'no_such_dataset'"):
        load("no_such_dataset")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestChecksummedDownloads:
    @pytest.fixture
    def published(self, tmp_path: Path) -> Path:
        path = tmp_path / "published" / "Checked.h5mu"
        path.parent.mkdir()
        build_synthetic_dataset(n_samples=4, n_drugs=3).save(UPath(path))
        return path

    @pytest.fixture
    def downloads(self, tmp_path: Path, published: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        cache = tmp_path / "cache"
        cache.mkdir()
        monkeypatch.setenv("DREVALPY_CACHE_DIR", str(cache))
        monkeypatch.setenv("DREVALPY_CONFIG_DIR", str(tmp_path / "config"))
        monkeypatch.setattr(load_module, "dataset_registry", Registry())
        calls: list[str] = []

        def fake_download(remote: object, local_path: UPath, label: str) -> UPath:
            calls.append(label)
            shutil.copyfile(published, local_path)
            return local_path

        monkeypatch.setattr(load_module, "download_file", fake_download)
        return calls

    def _register(self, tmp_path: Path, sha256: str | None) -> None:
        entry: dict = {"source": "src", "file": "Checked.h5mu"}
        if sha256 is not None:
            entry["sha256"] = sha256
        config_path = tmp_path / "config" / "datasets.json"
        config_path.parent.mkdir(exist_ok=True)
        config_path.write_text(
            json.dumps({"sources": {"src": "https://example.org/data"}, "datasets": {"Checked": entry}})
        )
        load_module.dataset_registry.reload()

    def test_file_sha256_matches_hashlib_is_remembered_and_follows_a_replaced_file(
        self, tmp_path: Path, published: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = UPath(published)
        assert _file_sha256(path) == _sha256(published)
        with monkeypatch.context() as patched:
            patched.setattr(UPath, "open", None)
            assert _file_sha256(path) == _sha256(published)

        changing = UPath(tmp_path) / "f.bin"
        changing.write_bytes(b"one")
        first = _file_sha256(changing)
        changing.write_bytes(b"three")
        assert _file_sha256(changing) != first

    def test_a_download_with_the_right_hash_is_kept_and_loaded(
        self, tmp_path: Path, published: Path, downloads: list[str]
    ) -> None:
        self._register(tmp_path, _sha256(published))

        assert load("Checked") is not None
        assert downloads == ["Checked"]
        assert (tmp_path / "cache" / "Checked.h5mu").is_file()

    def test_a_download_with_the_wrong_hash_raises_and_is_deleted(
        self, tmp_path: Path, published: Path, downloads: list[str]
    ) -> None:
        wrong = "0" * 64
        self._register(tmp_path, wrong)

        with pytest.raises(InvalidInputError) as excinfo:
            load("Checked")

        message = str(excinfo.value)
        assert "https://example.org/data/Checked.h5mu" in message
        assert wrong in message
        assert _sha256(published) in message
        assert not (tmp_path / "cache" / "Checked.h5mu").exists()

    def test_a_cached_file_with_the_right_hash_is_used_without_downloading(
        self, tmp_path: Path, published: Path, downloads: list[str]
    ) -> None:
        shutil.copyfile(published, tmp_path / "cache" / "Checked.h5mu")
        self._register(tmp_path, _sha256(published))

        assert load("Checked") is not None
        assert downloads == []

    def test_a_cached_file_with_another_hash_is_replaced(
        self, tmp_path: Path, published: Path, downloads: list[str], caplog: pytest.LogCaptureFixture
    ) -> None:
        old = tmp_path / "old.h5mu"
        build_synthetic_dataset(n_samples=5, n_drugs=2).save(UPath(old))
        shutil.copyfile(old, tmp_path / "cache" / "Checked.h5mu")
        self._register(tmp_path, _sha256(published))

        with caplog.at_level("INFO"):
            load("Checked")

        assert downloads == ["Checked"]
        assert _sha256(tmp_path / "cache" / "Checked.h5mu") == _sha256(published)
        assert "does not match the registered sha256" in caplog.text

    def test_without_a_hash_a_cached_file_is_taken_as_it_is_and_a_download_is_not_verified(
        self, tmp_path: Path, published: Path, downloads: list[str]
    ) -> None:
        old = tmp_path / "old.h5mu"
        build_synthetic_dataset(n_samples=5, n_drugs=2).save(UPath(old))
        shutil.copyfile(old, tmp_path / "cache" / "Checked.h5mu")
        self._register(tmp_path, None)

        assert load("Checked") is not None
        assert downloads == []
        assert _sha256(tmp_path / "cache" / "Checked.h5mu") == _sha256(old)

        (tmp_path / "cache" / "Checked.h5mu").unlink()
        assert load("Checked") is not None
        assert downloads == ["Checked"]

    def test_the_hash_check_precedes_parsing(self, tmp_path: Path, published: Path, downloads: list[str]) -> None:
        (tmp_path / "cache" / "Checked.h5mu").write_bytes(b"not an h5mu file")
        self._register(tmp_path, _sha256(published))

        assert load("Checked") is not None
        assert downloads == ["Checked"]

    def test_load_from_cache_checks_the_hash_case_insensitively(self, tmp_path: Path) -> None:
        path = UPath(tmp_path) / "f.h5mu"
        build_synthetic_dataset(n_samples=4, n_drugs=3).save(path)

        assert _load_from_cache(path, _sha256(Path(path)).upper()) is not None
        assert _load_from_cache(path, "0" * 64) is None
        assert not path.is_file()
