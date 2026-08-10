"""Smoke test mirror for superfeltr predictor package."""

from __future__ import annotations

from drevalpy.components.predictors.literature.superfeltr.predictor import SuperFELTRPredictor
from drevalpy.components.registry import get_predictor, register_builtins


def test_superfeltr_predictor_registry_name() -> None:
    register_builtins.ensure_predictor_registered("superfeltr")
    assert get_predictor("superfeltr") is SuperFELTRPredictor
