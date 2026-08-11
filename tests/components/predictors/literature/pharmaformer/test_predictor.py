"""Smoke test mirror for pharmaformer predictor package."""

from __future__ import annotations

from drevalpy.components.predictors.literature.pharmaformer.predictor import PharmaFormerPredictor
from drevalpy.registry._builtins import ensure_predictor_registered
from drevalpy.registry.predictor import get as get_predictor


def test_pharmaformer_predictor_registry_name() -> None:
    ensure_predictor_registered("pharmaFormer")
    assert get_predictor("pharmaFormer") is PharmaFormerPredictor
