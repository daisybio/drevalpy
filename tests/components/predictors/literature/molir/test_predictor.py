"""Smoke test mirror for molir predictor package."""

from __future__ import annotations

from drevalpy.components.predictors.literature.molir.predictor import MOLIRPredictor
from drevalpy.registry._builtins import ensure_predictor_registered
from drevalpy.registry.predictor import get as get_predictor


def test_molir_predictor_registry_name() -> None:
    ensure_predictor_registered("molir")
    assert get_predictor("molir") is MOLIRPredictor
