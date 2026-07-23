"""Smoke test mirror for superfeltr_predictor module."""

from __future__ import annotations

import drevalpy.components.register_builtins as register_builtins
from drevalpy.components.predictors.literature.superfeltr_predictor import SuperFELTRPredictor
from drevalpy.components.registry import get_predictor


def test_superfeltr_predictor_registry_name() -> None:
    register_builtins.ensure_predictor_registered("superfeltr")
    assert get_predictor("superfeltr") is SuperFELTRPredictor
