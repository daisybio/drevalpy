"""Smoke test mirror for pharmaformer predictor package."""

from __future__ import annotations

from drevalpy.components.predictors.literature.pharmaformer.predictor import PharmaFormerPredictor
from drevalpy.components.registry import get_predictor, register_builtins


def test_pharmaformer_predictor_registry_name() -> None:
    register_builtins.ensure_predictor_registered("pharmaFormer")
    assert get_predictor("pharmaFormer") is PharmaFormerPredictor
