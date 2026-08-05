"""Tests for shared registry base helpers."""

from __future__ import annotations

from drevalpy.components.registry.base import Registry
from drevalpy.components.registry.featurizer_registry import FeaturizerRegistry
from drevalpy.components.registry.predictor_registry import PredictorRegistry


def test_required_fields_are_explicit_per_registry() -> None:
    assert Registry._required_fields == ("description",)
    assert FeaturizerRegistry._required_fields == ("description", "contract")
    assert PredictorRegistry._required_fields == (
        "description",
        "cell_line_contract",
        "drug_contract",
    )
