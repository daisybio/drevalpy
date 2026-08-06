"""Tests for compact predictor config parsing."""

from __future__ import annotations

import pytest

from drevalpy.models.config._predictor_parse import normalize_predictor_config


def test_normalize_string_shorthand() -> None:
    payload = normalize_predictor_config("randomForest")
    assert payload == {"name": "randomForest"}


def test_normalize_one_key_mapping() -> None:
    payload = normalize_predictor_config({"randomForest": {"n_estimators": 10}})
    assert payload["name"] == "randomForest"
    assert payload["hyperparameter_space"]["n_estimators"]["default"] == 10


def test_normalize_rejects_invalid_shape() -> None:
    with pytest.raises(TypeError, match="string or mapping"):
        normalize_predictor_config(123)
