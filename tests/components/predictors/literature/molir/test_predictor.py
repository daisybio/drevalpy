"""Smoke test mirror for molir predictor package."""

from __future__ import annotations

from drevalpy.components.predictors.literature.molir.predictor import MOLIRPredictor
from drevalpy.components.registry import get_predictor, register_builtins


def test_molir_predictor_registry_name() -> None:
    register_builtins.ensure_predictor_registered("molir")
    assert get_predictor("molir") is MOLIRPredictor
