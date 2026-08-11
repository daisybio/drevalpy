"""Smoke test mirror for druggnn predictor package."""

from __future__ import annotations

from drevalpy.components.predictors.literature.druggnn.predictor import DrugGNNPredictor
from drevalpy.registry._builtins import ensure_predictor_registered
from drevalpy.registry.predictor import get as get_predictor


def test_druggnn_predictor_registry_name() -> None:
    ensure_predictor_registered("drugGNN")
    assert get_predictor("drugGNN") is DrugGNNPredictor
