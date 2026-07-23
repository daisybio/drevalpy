"""Smoke test mirror for molir_predictor module."""

from __future__ import annotations

import drevalpy.components.register_builtins as register_builtins
from drevalpy.components.predictors.literature.molir_predictor import MOLIRPredictor
from drevalpy.components.registry import get_predictor


def test_molir_predictor_registry_name() -> None:
    register_builtins.ensure_predictor_registered("molir")
    assert get_predictor("molir") is MOLIRPredictor
