"""Smoke test mirror for precily_predictor module."""

from __future__ import annotations

import drevalpy.components.register_builtins as register_builtins
from drevalpy.components.predictors.literature.precily_predictor import PrecilyPredictor
from drevalpy.components.registry import get_predictor


def test_precily_predictor_registry_name() -> None:
    register_builtins.ensure_predictor_registered("precily")
    assert get_predictor("precily") is PrecilyPredictor
