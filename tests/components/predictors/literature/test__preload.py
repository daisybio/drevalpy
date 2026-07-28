"""Smoke tests for literature preload helpers."""

from __future__ import annotations

from drevalpy.components.predictors.literature._preload import (
    DISCOVERED_HYPERPARAMETERS_KEY,
    merge_preload_hyperparameters,
)


def test_merge_preload_applies_discovered_hyperparameters() -> None:
    merged, remaining = merge_preload_hyperparameters(
        {"alpha": 1.0},
        {DISCOVERED_HYPERPARAMETERS_KEY: {"drug_dim": 64}, "layer_connections": [1, 2, 3]},
    )
    assert merged["drug_dim"] == 64
    assert merged["alpha"] == 1.0
    assert "layer_connections" in remaining
