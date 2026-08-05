"""Tests for featurizer registry types and singletons."""

from __future__ import annotations

from drevalpy.components.contracts import FeatureContract, FeatureFormat
from drevalpy.components.registry.featurizer import (
    CellLineFeaturizerRegistry,
    DrugFeaturizerRegistry,
    cell_line_featurizer_registry,
    drug_featurizer_registry,
)


def test_concrete_featurizer_registries_use_fixed_identity() -> None:
    cell = CellLineFeaturizerRegistry()
    drug = DrugFeaturizerRegistry()
    assert cell._registry_id == "cell_line_featurizer"
    assert drug._registry_id == "drug_featurizer"
    assert cell_line_featurizer_registry._display_name == "cell_line_featurizers"
    assert drug_featurizer_registry._display_name == "drug_featurizers"


def test_isolated_cell_line_registry_registers_with_contract() -> None:
    registry = CellLineFeaturizerRegistry()

    @registry.register("localFeat", description="local", contract=FeatureFormat.GRAPH)
    class LocalFeat:
        pass

    assert registry.get("localFeat") is LocalFeat
    assert vars(LocalFeat)["contract"] == FeatureContract(format=FeatureFormat.GRAPH)
