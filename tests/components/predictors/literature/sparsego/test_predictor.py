"""Smoke test mirror for sparsego predictor package."""

from __future__ import annotations

import drevalpy.components.register_builtins as register_builtins
from drevalpy.components.predictors.literature.sparsego.predictor import SparseGOPredictor
from drevalpy.components.registry import get_predictor


def test_sparsego_predictor_registry_name() -> None:
    register_builtins.ensure_predictor_registered("sparsego")
    assert get_predictor("sparsego") is SparseGOPredictor
