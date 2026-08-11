"""Smoke test mirror for sparsego predictor package."""

from __future__ import annotations

from drevalpy.components.predictors.literature.sparsego.predictor import SparseGOPredictor
from drevalpy.registry._builtins import ensure_predictor_registered
from drevalpy.registry.predictor import get as get_predictor


def test_sparsego_predictor_registry_name() -> None:
    ensure_predictor_registered("sparsego")
    assert get_predictor("sparsego") is SparseGOPredictor
