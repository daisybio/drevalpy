"""Smoke test mirror for dipk predictor package."""

from __future__ import annotations

from drevalpy.components.predictors.literature.dipk.predictor import DIPKPredictor
from drevalpy.components.registry import get_predictor, register_builtins


def test_dipk_predictor_registry_name() -> None:
    register_builtins.ensure_predictor_registered("dipk")
    assert get_predictor("dipk") is DIPKPredictor
