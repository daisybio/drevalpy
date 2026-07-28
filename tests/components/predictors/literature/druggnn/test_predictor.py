"""Smoke test mirror for druggnn predictor package."""

from __future__ import annotations

import drevalpy.components.register_builtins as register_builtins
from drevalpy.components.predictors.literature.druggnn.predictor import DrugGNNPredictor
from drevalpy.components.registry import get_predictor


def test_druggnn_predictor_registry_name() -> None:
    register_builtins.ensure_predictor_registered("drugGNN")
    assert get_predictor("drugGNN") is DrugGNNPredictor
