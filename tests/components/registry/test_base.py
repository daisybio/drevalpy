"""Tests for shared registry base helpers."""

from __future__ import annotations

from drevalpy.registry.components import ComponentRegistry
from drevalpy.registry.featurizer import FeaturizerRegistry
from drevalpy.registry.predictor import PredictorRegistry


def test_required_fields_are_explicit_per_registry() -> None:
    assert ComponentRegistry._required_fields == ("description",)
    assert FeaturizerRegistry._required_fields == ("description", "contract")
    assert PredictorRegistry._required_fields == (
        "description",
        "cell_line_contract",
        "drug_contract",
    )
