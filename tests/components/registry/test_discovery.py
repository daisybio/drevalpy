"""Tests for built-in discovery helper used by list_* APIs."""

from __future__ import annotations

from drevalpy.components.registry._discovery import ensure_builtins_for_discovery
from drevalpy.components.registry.featurizer_registry import cell_line_featurizer_registry
from drevalpy.components.registry.predictor_registry import predictor_registry


def test_ensure_builtins_for_discovery_registers_catalog() -> None:
    cell_line_featurizer_registry.clear()
    predictor_registry.clear()
    assert cell_line_featurizer_registry.list_names() == []
    assert predictor_registry.list_names() == []

    ensure_builtins_for_discovery()

    assert len(cell_line_featurizer_registry.list_names()) >= 1
    assert len(predictor_registry.list_names()) >= 1
