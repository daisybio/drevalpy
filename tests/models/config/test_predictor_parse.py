"""Tests for predictor parse."""

from __future__ import annotations

import pytest

from drevalpy.models.config._predictor_parse import normalize_predictor_config

SPACE = {"n_estimators": {"type": "int", "low": 2, "high": 20, "default": 4}}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("randomForest", {"name": "randomForest"}),
        ({"randomForest": None}, {"name": "randomForest"}),
        (
            {"name": "randomForest", "hyperparameter_space": SPACE},
            {"name": "randomForest", "hyperparameter_space": SPACE},
        ),
    ],
)
def test_normalize_shapes(raw: object, expected: dict) -> None:
    assert normalize_predictor_config(raw) == expected


@pytest.mark.parametrize(
    ("raw", "default"),
    [
        ({"randomForest": {"n_estimators": 10}}, 10),
        ({"randomForest": {"hyperparameter_space": SPACE, "n_estimators": 10}}, 4),
    ],
)
def test_loose_value_becomes_default_unless_space_given(raw: dict, default: int) -> None:
    payload = normalize_predictor_config(raw)
    assert payload["hyperparameter_space"]["n_estimators"]["default"] == default


@pytest.mark.parametrize(
    ("raw", "error", "match"),
    [
        ({"randomForest": {"not_a_knob": 1}}, ValueError, "do not accept non-tunable options \\('not_a_knob'\\)"),
        ({"randomForest": 5}, ValueError, "must be a mapping when provided"),
        (123, TypeError, "string or mapping"),
        ({"hyperparameter_space": {}, "extra": 1}, ValueError, "string, one-key mapping, or dict with 'name'"),
    ],
)
def test_invalid_shapes_rejected(raw: object, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        normalize_predictor_config(raw)
