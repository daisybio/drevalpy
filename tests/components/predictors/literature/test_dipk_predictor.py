"""Smoke test mirror for dipk_predictor module."""

from __future__ import annotations

import drevalpy.components.register_builtins as register_builtins
from drevalpy.components.predictors.literature.dipk_predictor import DIPKPredictor
from drevalpy.components.registry import get_predictor


def test_dipk_predictor_registry_name() -> None:
    register_builtins.ensure_predictor_registered("dipk")
    assert get_predictor("dipk") is DIPKPredictor
