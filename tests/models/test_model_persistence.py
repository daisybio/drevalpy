"""Tests for drevalpy-model checkpoint persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import pytest

from drevalpy.models import construct_model
from drevalpy.models._model_persistence import (
    FORMAT_NAME,
    FORMAT_VERSION,
    STATE_FILE,
    CorruptedCheckpointError,
    UnsupportedCheckpointFormatError,
    load_model_payload,
    save_model,
)
from drevalpy.models.config import ModelConfig
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


def test_round_trip_save_load() -> None:
    model = _fitted_model()
    with tempfile.TemporaryDirectory() as directory:
        save_model(model, directory)
        loaded = construct_model("ElasticNet").load(directory)
    assert loaded._stack is not None
    assert loaded._stack.is_fitted()
    assert loaded._resolved_model_config is not None
    assert loaded._resolved_model_config.predictor.hyperparameters["alpha"] == 0.1


def test_load_missing_checkpoint_raises_file_not_found() -> None:
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(FileNotFoundError, match="Missing model checkpoint"):
            load_model_payload(directory)


def test_load_rejects_non_mapping_state() -> None:
    config = ModelConfig.from_spec("ElasticNet", hyperparameters={"alpha": 0.1, "l1_ratio": 0.5})
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "model_name": "ElasticNet",
        "config": config.model_dump(mode="json"),
        "state": "bad",
    }
    with tempfile.TemporaryDirectory() as directory:
        joblib.dump(payload, Path(directory) / STATE_FILE)
        with pytest.raises(CorruptedCheckpointError, match="checkpoint state is not a mapping"):
            load_model_payload(directory)


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
        path = Path(directory) / STATE_FILE
        joblib.dump(payload, path)
        with pytest.raises(error_type, match=match):
            load_model_payload(directory)


def test_load_rejects_unfitted_checkpoint_state() -> None:
    config = ModelConfig.from_spec("ElasticNet", hyperparameters={"alpha": 0.1, "l1_ratio": 0.5})
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "model_name": "ElasticNet",
        "config": config.model_dump(mode="json"),
        "state": {"predictor": {}, "cell_line_featurizer": {}, "drug_featurizer": {}},
    }
    with tempfile.TemporaryDirectory() as directory:
        joblib.dump(payload, Path(directory) / STATE_FILE)
        with pytest.raises(
            CorruptedCheckpointError, match="missing a fitted estimator|did not restore a fitted predictor"
        ):
            construct_model("ElasticNet").load(directory)
