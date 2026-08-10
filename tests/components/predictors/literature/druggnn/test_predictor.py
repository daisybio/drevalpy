"""Smoke test mirror for druggnn predictor package."""

from __future__ import annotations

from drevalpy.components.predictors.literature.druggnn.predictor import DrugGNNPredictor
from drevalpy.components.registry import get_predictor, register_builtins


def test_druggnn_predictor_registry_name() -> None:
    register_builtins.ensure_predictor_registered("drugGNN")
    assert get_predictor("drugGNN") is DrugGNNPredictor
