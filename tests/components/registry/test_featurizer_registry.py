"""Tests for FeaturizerRegistry type and singletons."""

from __future__ import annotations

from drevalpy.components.contracts import FeatureContract, FeatureFormat
from drevalpy.components.registry.featurizer_registry import (
    FeaturizerRegistry,
    cell_line_featurizer_registry,
    drug_featurizer_registry,
)


def test_module_featurizer_singletons_use_fixed_identity() -> None:
    assert cell_line_featurizer_registry._registry_id == "cell_line_featurizer"
    assert drug_featurizer_registry._registry_id == "drug_featurizer"
    assert cell_line_featurizer_registry._display_name == "cell_line_featurizers"
    assert drug_featurizer_registry._display_name == "drug_featurizers"


def test_isolated_featurizer_registry_registers_with_contract() -> None:
    registry = FeaturizerRegistry("test", "Test featurizer", "test_featurizers")

    @registry.register("localFeat", description="local", contract=FeatureFormat.GRAPH)
    class LocalFeat:
        pass

    assert registry.get("localFeat") is LocalFeat
    assert vars(LocalFeat)["contract"] == FeatureContract(format=FeatureFormat.GRAPH)


def test_register_existing_restores_registry_name() -> None:
    registry = FeaturizerRegistry("test", "Test component", "test_components")
    decorated = registry.register(
        "restored",
        description="restored",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )

    @decorated
    class Restored:
        pass

    assert vars(Restored)["registry_name"] == "restored"
    registry.clear()
    assert registry.list_names() == []

    registry.register_existing("restored", Restored)
    assert registry.get("restored") is Restored
    assert vars(Restored)["registry_name"] == "restored"


def test_registry_clear() -> None:
    registry = FeaturizerRegistry("test", "Test component", "test_components")
    decorated = registry.register("x", description="x", contract=FeatureFormat.NUMERIC_MATRIX)

    @decorated
    class X:
        pass

    assert registry.list_names() == ["x"]
    registry.clear()
    assert registry.list_names() == []
