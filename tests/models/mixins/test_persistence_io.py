"""Tests for low-level checkpoint archive I/O.

Covers ``drevalpy.models.mixins._persistence_io``: archive writing, payload
validation and path resolution. The ``DRPPersistenceMixin`` wrapper that calls
into it is tested in ``test_persistence.py``.
"""

from __future__ import annotations

import zipfile

import pytest
from upath import UPath

from drevalpy.models import construct_model, load_model
from drevalpy.models.mixins._persistence_io import (
    FORMAT_NAME,
    FORMAT_VERSION,
    PAYLOAD_MEMBER,
    CorruptedCheckpointError,
    IncompatibleModelCheckpointError,
    ModelCheckpointError,
    UnsupportedCheckpointFormatError,
    load_model_payload,
    resolve_checkpoint_path,
    save_model,
)
from tests.models.mixins._helpers import elastic_net_payload, fitted_elastic_net, write_archive
from tests.models.synthetic_fixtures import (
    lco_split_masks,
    synthetic_mudataset_gene_expression_fingerprints,
)


class TestResolveCheckpointPath:
    """Archive path normalization."""

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
    """Writing a fitted model to an archive."""

    def test_appends_the_zip_suffix(self, tmp_path) -> None:
        model = fitted_elastic_net()

        save_model(model, str(UPath(tmp_path) / "elastic_net"))

        assert (UPath(tmp_path) / "elastic_net.zip").is_file()

    def test_writes_the_payload_member(self, tmp_path) -> None:
        model = fitted_elastic_net()
        archive = UPath(tmp_path) / "elastic_net.zip"

        save_model(model, str(archive))

        with zipfile.ZipFile(archive) as handle:
            assert handle.namelist() == [PAYLOAD_MEMBER]

    def test_leaves_no_temporary_files(self, tmp_path) -> None:
        model = fitted_elastic_net()

        save_model(model, str(UPath(tmp_path) / "elastic_net"))

        assert [path.name for path in UPath(tmp_path).iterdir()] == ["elastic_net.zip"]

    def test_rejects_an_existing_directory(self, tmp_path) -> None:
        model = fitted_elastic_net()

        with pytest.raises(ValueError, match="not a directory"):
            save_model(model, str(tmp_path))

    def test_rejects_an_untrained_model(self, tmp_path) -> None:
        model = construct_model("ElasticNet")({"alpha": 0.1, "l1_ratio": 0.5})

        with pytest.raises(RuntimeError, match="not trained"):
            save_model(model, str(UPath(tmp_path) / "elastic_net"))


class TestLoadModelPayload:
    """Reading and validating a checkpoint payload."""

    def test_round_trips_model_identity_and_config(self, tmp_path) -> None:
        model = fitted_elastic_net()
        archive = UPath(tmp_path) / "elastic_net.zip"
        save_model(model, str(archive))

        model_name, config, state = load_model_payload(str(archive))

        assert model_name == "ElasticNet"
        assert config.predictor_values()["alpha"] == 0.1
        assert set(state) == {"predictor", "cell_line_featurizer", "drug_featurizer"}

    def test_accepts_a_path_without_the_zip_suffix(self, tmp_path) -> None:
        model = fitted_elastic_net()
        save_model(model, str(UPath(tmp_path) / "elastic_net"))

        model_name, _, _ = load_model_payload(str(UPath(tmp_path) / "elastic_net"))

        assert model_name == "ElasticNet"

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

    def test_rejects_a_non_mapping_state(self, tmp_path) -> None:
        archive = write_archive(UPath(tmp_path) / "elastic_net.zip", elastic_net_payload("bad"))

        with pytest.raises(CorruptedCheckpointError, match="checkpoint state is not a mapping"):
            load_model_payload(str(archive))

    @pytest.mark.parametrize(
        ("model_name", "match"),
        [
            pytest.param(None, "model_name is missing or invalid", id="missing-name"),
            pytest.param("", "model_name is missing or invalid", id="empty-name"),
        ],
    )
    def test_rejects_a_missing_model_name(self, tmp_path, model_name: str | None, match: str) -> None:
        payload = elastic_net_payload({}, model_name=model_name)  # type: ignore[arg-type]
        archive = write_archive(UPath(tmp_path) / "elastic_net.zip", payload)

        with pytest.raises(CorruptedCheckpointError, match=match):
            load_model_payload(str(archive))

    @pytest.mark.parametrize(
        ("payload", "error_type", "match"),
        [
            pytest.param("not-a-mapping", CorruptedCheckpointError, "not a mapping", id="payload-not-a-mapping"),
            pytest.param(
                {"format": "unknown-format", "version": 0, "model_name": "ElasticNet", "config": {}, "state": {}},
                UnsupportedCheckpointFormatError,
                "unsupported checkpoint format/version",
                id="unknown-format",
            ),
            pytest.param(
                {"format": FORMAT_NAME, "version": 1, "model_name": "ElasticNet", "config": {}, "state": {}},
                UnsupportedCheckpointFormatError,
                "unsupported checkpoint format/version",
                id="older-version",
            ),
            pytest.param(
                {
                    "format": FORMAT_NAME,
                    "version": FORMAT_VERSION + 1,
                    "model_name": "ElasticNet",
                    "config": {},
                    "state": {},
                },
                UnsupportedCheckpointFormatError,
                "unsupported checkpoint format/version",
                id="newer-version",
            ),
            pytest.param(
                {
                    "format": FORMAT_NAME,
                    "version": FORMAT_VERSION,
                    "model_name": "ElasticNet",
                    "config": "bad",
                    "state": {},
                },
                CorruptedCheckpointError,
                "checkpoint config is invalid",
                id="unparsable-config",
            ),
        ],
    )
    def test_rejects_malformed_or_unsupported_payloads(
        self,
        tmp_path,
        payload: object,
        error_type: type[Exception],
        match: str,
    ) -> None:
        archive = write_archive(UPath(tmp_path) / "elastic_net.zip", payload)

        with pytest.raises(error_type, match=match):
            load_model_payload(str(archive))


class TestLoadModel:
    """Reconstructing a model without a class handle."""

    def test_reconstructs_from_the_stored_model_name(self, tmp_path) -> None:
        model = fitted_elastic_net()
        checkpoint = str(UPath(tmp_path) / "elastic_net")
        save_model(model, checkpoint)

        loaded = load_model(checkpoint)

        assert loaded.get_model_name() == "ElasticNet"
        assert loaded._stack is not None
        assert loaded._stack.is_fitted()

    def test_restores_the_hyperparameters(self, tmp_path) -> None:
        model = fitted_elastic_net()
        checkpoint = str(UPath(tmp_path) / "elastic_net")
        save_model(model, checkpoint)

        loaded = load_model(checkpoint)

        assert loaded._resolved_model_config is not None
        assert loaded._resolved_model_config.predictor_values()["alpha"] == 0.1

    def test_accepts_an_explicit_zip_path(self, tmp_path) -> None:
        model = fitted_elastic_net()
        archive = str(UPath(tmp_path) / "elastic_net.zip")
        save_model(model, archive)

        loaded = load_model(archive)

        assert loaded.get_model_name() == "ElasticNet"

    def test_supports_custom_model_names(self, tmp_path) -> None:
        model = construct_model("MyRF", "scaledGeneExpression:fingerprints:randomForest")({"n_estimators": 5})
        model.train(synthetic_mudataset_gene_expression_fingerprints(), lco_split_masks())
        checkpoint = str(UPath(tmp_path) / "my_rf")
        save_model(model, checkpoint)

        loaded = load_model(checkpoint)

        assert loaded.get_model_name() == "MyRF"
        assert loaded._stack is not None
        assert loaded._stack.is_fitted()


class TestErrorHierarchy:
    """All checkpoint errors are catchable as one family and as ``ValueError``."""

    @pytest.mark.parametrize(
        "error_type",
        [
            pytest.param(UnsupportedCheckpointFormatError, id="unsupported-format"),
            pytest.param(CorruptedCheckpointError, id="corrupted"),
            pytest.param(IncompatibleModelCheckpointError, id="incompatible"),
        ],
    )
    def test_subclasses_model_checkpoint_error_and_value_error(self, error_type: type[Exception]) -> None:
        assert issubclass(error_type, ModelCheckpointError)
        assert issubclass(error_type, ValueError)

    def test_base_error_is_not_a_value_error(self) -> None:
        assert not issubclass(ModelCheckpointError, ValueError)
