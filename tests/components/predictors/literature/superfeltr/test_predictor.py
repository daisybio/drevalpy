"""Smoke test mirror for superfeltr predictor package."""

from __future__ import annotations

from drevalpy.components.predictors.literature.superfeltr.predictor import SuperFELTRPredictor
from drevalpy.registry._builtins import ensure_predictor_registered
from drevalpy.registry.predictor import get as get_predictor


def test_superfeltr_predictor_registry_name() -> None:
    ensure_predictor_registered("superfeltr")
    assert get_predictor("superfeltr") is SuperFELTRPredictor
