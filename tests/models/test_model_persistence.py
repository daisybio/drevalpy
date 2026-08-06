"""Tests for drevalpy-model checkpoint persistence."""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import joblib
import pytest

from drevalpy.models import construct_model, load_model
from drevalpy.models._model_persistence import (
    FORMAT_NAME,
    FORMAT_VERSION,
    PAYLOAD_MEMBER,
    CorruptedCheckpointError,
    UnsupportedCheckpointFormatError,
    load_model_payload,
    resolve_checkpoint_path,
    save_model,
)
from drevalpy.models.config import from_spec
from tests.models.synthetic_fixtures import (
    cell_line_gene_expression,
    drug_fingerprints,
    multi_drug_response,
)


def _fitted_model():
    model = construct_model("ElasticNet")({"alpha": 0.1, "l1_ratio": 0.5})
    response = multi_drug_response()
    model.train(response, cell_line_gene_expression(), drug_fingerprints())
    return model


def _write_archive(archive_path: Path, payload: object) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    joblib.dump(payload, buffer)
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(PAYLOAD_MEMBER, buffer.getvalue())
    return archive_path


def test_round_trip_save_load() -> None:
    model = _fitted_model()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = str(Path(directory) / "elastic_net")
        save_model(model, checkpoint)
        assert Path(f"{checkpoint}.zip").is_file()
        loaded = construct_model("ElasticNet").load(checkpoint)
    assert loaded._stack is not None
    assert loaded._stack.is_fitted()
    assert loaded._resolved_model_config is not None
    assert loaded._resolved_model_config.predictor_values()["alpha"] == 0.1


def test_round_trip_save_load_explicit_zip_path() -> None:
    model = _fitted_model()
    with tempfile.TemporaryDirectory() as directory:
        archive = str(Path(directory) / "elastic_net.zip")
        save_model(model, archive)
        assert Path(archive).is_file()
        loaded = load_model(archive)
    assert loaded.get_model_name() == "ElasticNet"
    assert loaded._stack is not None
    assert loaded._stack.is_fitted()


def test_load_model_reconstructs_without_class_handle() -> None:
    model = _fitted_model()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = str(Path(directory) / "elastic_net")
        model.save(checkpoint)
        loaded = load_model(checkpoint)
    assert loaded.get_model_name() == "ElasticNet"
    assert loaded._stack is not None
    assert loaded._stack.is_fitted()
    assert loaded._resolved_model_config is not None
    assert loaded._resolved_model_config.predictor_values()["alpha"] == 0.1


def test_load_model_supports_custom_model_names() -> None:
    model = construct_model("MyRF", "scaledGeneExpression:fingerprints:randomForest")({"n_estimators": 5})
    response = multi_drug_response()
    model.train(response, cell_line_gene_expression(), drug_fingerprints())
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = str(Path(directory) / "my_rf")
        model.save(checkpoint)
        loaded = load_model(checkpoint)
    assert loaded.get_model_name() == "MyRF"
    assert loaded._stack is not None
    assert loaded._stack.is_fitted()


def test_load_missing_checkpoint_raises_file_not_found() -> None:
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(FileNotFoundError, match="Missing model checkpoint"):
            load_model_payload(str(Path(directory) / "missing"))


def test_save_rejects_directory_path() -> None:
    model = _fitted_model()
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(ValueError, match="not a directory"):
            model.save(directory)


def test_load_rejects_non_mapping_state() -> None:
    config = from_spec("ElasticNet", hyperparameters={"alpha": 0.1, "l1_ratio": 0.5})
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "model_name": "ElasticNet",
        "config": config.model_dump(mode="json"),
        "state": "bad",
    }
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "elastic_net.zip"
        _write_archive(archive, payload)
        with pytest.raises(CorruptedCheckpointError, match="checkpoint state is not a mapping"):
            load_model_payload(str(archive))


def test_load_rejects_invalid_zip() -> None:
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "elastic_net.zip"
        archive.write_text("not a zip", encoding="utf-8")
        with pytest.raises(CorruptedCheckpointError, match="not a valid zip file"):
            load_model_payload(str(archive))


def test_resolve_checkpoint_path_appends_zip() -> None:
    assert resolve_checkpoint_path("checkpoints/foo") == Path("checkpoints/foo.zip")
    assert resolve_checkpoint_path("checkpoints/foo.zip") == Path("checkpoints/foo.zip")
    assert resolve_checkpoint_path("checkpoints/foo.dreval.zip") == Path("checkpoints/foo.dreval.zip")
    assert resolve_checkpoint_path("checkpoints/foo.ZIP") == Path("checkpoints/foo.ZIP")


@pytest.mark.parametrize(
    ("payload", "error_type", "match"),
    [
        ("not-a-mapping", CorruptedCheckpointError, "not a mapping"),
        (
            {"format": "legacy", "version": 0, "model_name": "ElasticNet", "config": {}, "state": {}},
            UnsupportedCheckpointFormatError,
            "unsupported checkpoint format/version",
        ),
        (
            {
                "format": FORMAT_NAME,
                "version": FORMAT_VERSION + 1,
                "model_name": "ElasticNet",
                "config": {},
                "state": {},
            },
            UnsupportedCheckpointFormatError,
            "unsupported checkpoint format/version",
        ),
        (
            {
                "format": FORMAT_NAME,
                "version": FORMAT_VERSION,
                "model_name": "ElasticNet",
                "config": "bad",
                "state": {},
            },
            CorruptedCheckpointError,
            "checkpoint config is invalid",
        ),
    ],
)
def test_load_rejects_malformed_or_unsupported_payloads(
    payload: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "elastic_net.zip"
        _write_archive(archive, payload)
        with pytest.raises(error_type, match=match):
            load_model_payload(str(archive))


def test_load_rejects_unfitted_checkpoint_state() -> None:
    config = from_spec("ElasticNet", hyperparameters={"alpha": 0.1, "l1_ratio": 0.5})
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "model_name": "ElasticNet",
        "config": config.model_dump(mode="json"),
        "state": {"predictor": {}, "cell_line_featurizer": {}, "drug_featurizer": {}},
    }
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "elastic_net.zip"
        _write_archive(archive, payload)
        with pytest.raises(
            CorruptedCheckpointError, match="missing a fitted estimator|did not restore a fitted predictor"
        ):
            construct_model("ElasticNet").load(str(archive))
