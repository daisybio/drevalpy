"""Tests for native composed-model checkpoint persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import pytest

from drevalpy.models._component_persistence import (
    FORMAT_NAME,
    FORMAT_VERSION,
    STATE_FILE,
    CorruptedCheckpointError,
    UnsupportedCheckpointFormatError,
    load_composed_model,
    save_composed_model,
)
from drevalpy.models.config import ModelConfig
from tests.models.synthetic_fixtures import (
    cell_line_gene_expression,
    drug_fingerprints,
    multi_drug_response,
)


def _fitted_model():
    model = ModelConfig.from_spec("ElasticNet", hyperparameters={"alpha": 0.1, "l1_ratio": 0.5}).create_model()
    response = multi_drug_response()
    model.train(response, cell_line_gene_expression(), drug_fingerprints())
    return model


def test_round_trip_save_load() -> None:
    model = _fitted_model()
    with tempfile.TemporaryDirectory() as directory:
        save_composed_model(model, directory)
        loaded = load_composed_model(directory)
    assert loaded.is_fitted()
    assert loaded.config is not None
    assert loaded.config.predictor.hyperparameters["alpha"] == 0.1


def test_load_missing_checkpoint_raises_file_not_found() -> None:
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(FileNotFoundError, match="Missing native composed-model checkpoint"):
            load_composed_model(directory)


def test_load_rejects_non_mapping_state() -> None:
    config = ModelConfig.from_spec("ElasticNet", hyperparameters={"alpha": 0.1, "l1_ratio": 0.5})
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "config": config.model_dump(mode="json"),
        "state": "bad",
    }
    with tempfile.TemporaryDirectory() as directory:
        joblib.dump(payload, Path(directory) / STATE_FILE)
        with pytest.raises(CorruptedCheckpointError, match="checkpoint state is not a mapping"):
            load_composed_model(directory)


@pytest.mark.parametrize(
    ("payload", "error_type", "match"),
    [
        ("not-a-mapping", CorruptedCheckpointError, "not a mapping"),
        (
            {"format": "legacy", "version": 0, "config": {}, "state": {}},
            UnsupportedCheckpointFormatError,
            "unsupported checkpoint format/version",
        ),
        (
            {"format": FORMAT_NAME, "version": FORMAT_VERSION + 1, "config": {}, "state": {}},
            UnsupportedCheckpointFormatError,
            "unsupported checkpoint format/version",
        ),
        (
            {"format": FORMAT_NAME, "version": FORMAT_VERSION, "config": "bad", "state": {}},
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
            load_composed_model(directory)


def test_load_rejects_unfitted_checkpoint_state() -> None:
    config = ModelConfig.from_spec("ElasticNet", hyperparameters={"alpha": 0.1, "l1_ratio": 0.5})
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "config": config.model_dump(mode="json"),
        "state": {"predictor": {}, "cell_line_featurizer": {}, "drug_featurizer": {}},
    }
    with tempfile.TemporaryDirectory() as directory:
        joblib.dump(payload, Path(directory) / STATE_FILE)
        with pytest.raises(
            CorruptedCheckpointError, match="missing a fitted estimator|did not restore a fitted predictor"
        ):
            load_composed_model(directory)
