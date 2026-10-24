"""Tests for low-level checkpoint archive I/O (``drevalpy.models.mixins._persistence_io``)."""

from __future__ import annotations

import shutil
import tempfile
import zipfile

import pytest
from upath import UPath

from drevalpy.errors import DrEvalError
from drevalpy.models import DRPModel, ModelSpec, load_model
from drevalpy.models.mixins._persistence_io import (
    FORMAT_VERSION,
    PAYLOAD_MEMBER,
    CorruptedCheckpointError,
    ModelCheckpointError,
    UnsupportedCheckpointFormatError,
    load_model_payload,
    resolve_checkpoint_path,
    save_model,
)
from tests.models.mixins._helpers import elastic_net_payload, fitted_elastic_net, write_archive
from tests.models.synthetic_fixtures import (
    lso_split_masks,
    small_data_spec,
    synthetic_mudataset_gene_expression_fingerprints,
)


class TestResolveCheckpointPath:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            pytest.param("checkpoints/foo", "checkpoints/foo.zip", id="suffix-appended"),
            pytest.param("checkpoints/foo.zip", "checkpoints/foo.zip", id="suffix-kept"),
            pytest.param("checkpoints/foo.dreval.zip", "checkpoints/foo.dreval.zip", id="compound-suffix-kept"),
            pytest.param("checkpoints/foo.ZIP", "checkpoints/foo.ZIP", id="suffix-match-is-case-insensitive"),
        ],
    )
    def test_resolves_to_an_archive_path(self, given: str, expected: str) -> None:
        assert resolve_checkpoint_path(given) == UPath(expected)

    def test_rejects_a_directory_style_path(self) -> None:
        with pytest.raises(ValueError, match="not a directory"):
            resolve_checkpoint_path("checkpoints/")


class TestSaveModel:
    def test_writes_a_single_payload_archive_and_leaves_no_temporary_files(self, tmp_path) -> None:
        save_model(fitted_elastic_net(), str(UPath(tmp_path) / "elastic_net"))

        assert [path.name for path in UPath(tmp_path).iterdir()] == ["elastic_net.zip"]
        with zipfile.ZipFile(UPath(tmp_path) / "elastic_net.zip") as handle:
            assert handle.namelist() == [PAYLOAD_MEMBER]

    def test_rejects_an_existing_directory(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="not a directory"):
            save_model(fitted_elastic_net(), str(tmp_path))

    def test_rejects_an_untrained_model(self, tmp_path) -> None:
        model = DRPModel(small_data_spec("ElasticNet"), {"alpha": 0.1, "l1_ratio": 0.5})

        with pytest.raises(RuntimeError, match="pipeline must be FITTED"):
            save_model(model, str(UPath(tmp_path) / "elastic_net"))


class TestLoadModelPayload:
    def test_round_trips_model_identity_and_config(self, tmp_path) -> None:
        save_model(fitted_elastic_net(), str(UPath(tmp_path) / "elastic_net"))

        model_name, config, state = load_model_payload(str(UPath(tmp_path) / "elastic_net"))

        assert model_name == "ElasticNet"
        assert config.predictor_values()["alpha"] == 0.1
        assert set(state) == {
            "predictor",
            "sample_featurizer",
            "drug_featurizer",
            "response_prototype",
            "response_transform",
        }

    def test_missing_archive_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="Missing model checkpoint"):
            load_model_payload(str(UPath(tmp_path) / "missing"))

    def test_rejects_a_non_zip_file(self, tmp_path) -> None:
        archive = UPath(tmp_path) / "elastic_net.zip"
        archive.write_text("not a zip", encoding="utf-8")

        with pytest.raises(CorruptedCheckpointError, match="not a valid zip file"):
            load_model_payload(str(archive))

    def test_rejects_an_archive_without_the_payload_member(self, tmp_path) -> None:
        archive = UPath(tmp_path) / "elastic_net.zip"
        with zipfile.ZipFile(archive, mode="w") as handle:
            handle.writestr("something-else", b"")

        with pytest.raises(CorruptedCheckpointError, match=f"missing {PAYLOAD_MEMBER!r}"):
            load_model_payload(str(archive))

    def test_rejects_another_version_and_says_how_to_recover(self, tmp_path) -> None:
        assert FORMAT_VERSION == 1
        payload = elastic_net_payload({})
        payload["version"] = 5
        archive = write_archive(UPath(tmp_path) / "elastic_net.zip", payload)

        with pytest.raises(UnsupportedCheckpointFormatError, match=r"unsupported checkpoint format/version") as info:
            load_model_payload(str(archive))
        assert f"version {FORMAT_VERSION}" in str(info.value)
        assert "retrain the model and save it again" in str(info.value)

    @pytest.mark.parametrize(
        ("payload", "error_type", "match"),
        [
            pytest.param("not-a-mapping", CorruptedCheckpointError, "not a mapping", id="payload-not-a-mapping"),
            pytest.param(
                elastic_net_payload("bad"), CorruptedCheckpointError, "checkpoint state is not a mapping", id="state"
            ),
            pytest.param(
                elastic_net_payload({}, model_name=None),  # type: ignore[arg-type]
                CorruptedCheckpointError,
                "model_name is missing or invalid",
                id="missing-name",
            ),
            pytest.param(
                elastic_net_payload({}, model_name=""),
                CorruptedCheckpointError,
                "model_name is missing or invalid",
                id="empty-name",
            ),
            pytest.param(
                {**elastic_net_payload({}), "format": "unknown-format"},
                UnsupportedCheckpointFormatError,
                "unsupported checkpoint format/version",
                id="unknown-format",
            ),
            pytest.param(
                {**elastic_net_payload({}), "version": 0},
                UnsupportedCheckpointFormatError,
                "unsupported checkpoint format/version",
                id="older-version",
            ),
            pytest.param(
                {**elastic_net_payload({}), "version": FORMAT_VERSION + 1},
                UnsupportedCheckpointFormatError,
                "unsupported checkpoint format/version",
                id="newer-version",
            ),
            pytest.param(
                {**elastic_net_payload({}), "config": "bad"},
                CorruptedCheckpointError,
                "checkpoint config is invalid",
                id="unparsable-config",
            ),
        ],
    )
    def test_rejects_malformed_or_unsupported_payloads(
        self, tmp_path, payload: object, error_type: type[Exception], match: str
    ) -> None:
        archive = write_archive(UPath(tmp_path) / "elastic_net.zip", payload)

        with pytest.raises(error_type, match=match):
            load_model_payload(str(archive))


class TestRemoteCheckpoints:
    def test_memory_round_trip(self) -> None:
        checkpoint = "memory://checkpoints/nested/elastic_net"

        save_model(fitted_elastic_net(), checkpoint)
        loaded = load_model(checkpoint)

        assert UPath("memory://checkpoints/nested/elastic_net.zip").is_file()
        assert loaded.name == "ElasticNet"
        assert loaded._pipeline.is_fitted()

    def test_remote_save_leaves_no_local_temporary_files(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

        save_model(fitted_elastic_net(), UPath("memory://checkpoints/elastic_net.zip"))

        assert list(UPath(tmp_path).iterdir()) == []

    def test_failed_remote_save_removes_the_local_temporary_file(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        monkeypatch.setattr(shutil, "copyfileobj", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("upload")))

        with pytest.raises(OSError, match="upload"):
            save_model(fitted_elastic_net(), UPath("memory://checkpoints/elastic_net.zip"))

        assert list(UPath(tmp_path).iterdir()) == []


class TestLoadModel:
    @pytest.mark.parametrize("suffix", ["", ".zip"])
    def test_reconstructs_from_the_stored_model_name_and_hyperparameters(self, tmp_path, suffix) -> None:
        checkpoint = str(UPath(tmp_path) / f"elastic_net{suffix}")
        save_model(fitted_elastic_net(), checkpoint)

        loaded = load_model(checkpoint)

        assert loaded.name == "ElasticNet"
        assert loaded._pipeline.is_fitted()
        assert loaded.config.predictor_values()["alpha"] == 0.1

    def test_supports_custom_model_names(self, tmp_path) -> None:
        from drevalpy.models.config import from_spec

        model = DRPModel(
            ModelSpec("MyRF", from_spec("scaledGeneExpression:fingerprints:randomForest")), {"n_estimators": 5}
        )
        model.train(synthetic_mudataset_gene_expression_fingerprints(), lso_split_masks().train)
        checkpoint = str(UPath(tmp_path) / "my_rf")
        save_model(model, checkpoint)

        loaded = load_model(checkpoint)

        assert loaded.name == "MyRF"
        assert loaded._pipeline.is_fitted()


class TestErrorHierarchy:
    @pytest.mark.parametrize("error_type", [UnsupportedCheckpointFormatError, CorruptedCheckpointError])
    def test_subclasses_model_checkpoint_error_and_value_error(self, error_type: type[Exception]) -> None:
        assert issubclass(error_type, ModelCheckpointError)
        assert issubclass(error_type, ValueError)

    def test_base_error_is_a_drevalpy_error_but_not_a_value_error(self) -> None:
        assert issubclass(ModelCheckpointError, DrEvalError)
        assert not issubclass(ModelCheckpointError, ValueError)
