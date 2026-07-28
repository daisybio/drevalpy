"""Smoke test mirror for srmf predictor package."""

from __future__ import annotations

import drevalpy.components.register_builtins as register_builtins
from drevalpy.components.predictors.literature.srmf.predictor import SRMFPredictor
from drevalpy.components.registry import get_predictor


def test_srmf_predictor_registry_name() -> None:
    register_builtins.ensure_predictor_registered("srmf")
    assert get_predictor("srmf") is SRMFPredictor
