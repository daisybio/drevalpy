"""Smoke test mirror for dipk predictor package."""

from __future__ import annotations

from drevalpy.components.predictors.literature.dipk.predictor import DIPKPredictor
from drevalpy.registry._builtins import ensure_predictor_registered
from drevalpy.registry.predictor import get as get_predictor


def test_dipk_predictor_registry_name() -> None:
    ensure_predictor_registered("dipk")
    assert get_predictor("dipk") is DIPKPredictor
