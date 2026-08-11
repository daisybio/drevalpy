"""Smoke test mirror for precily predictor package."""

from __future__ import annotations

from drevalpy.components.predictors.literature.precily.predictor import PrecilyPredictor
from drevalpy.registry._builtins import ensure_predictor_registered
from drevalpy.registry.predictor import get as get_predictor


def test_precily_predictor_registry_name() -> None:
    ensure_predictor_registered("precily")
    assert get_predictor("precily") is PrecilyPredictor
