"""Tests for CellLineFeaturizerRegistry type and singleton."""

from __future__ import annotations

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.registry.cell_line_featurizer import cell_line_featurizer_registry
from drevalpy.registry.cell_line_featurizer._registry import CellLineFeaturizerRegistry
from drevalpy.registry.featurizer import FeaturizerRegistry


def test_cell_line_featurizer_registry_subclasses_the_shared_base() -> None:
    assert issubclass(CellLineFeaturizerRegistry, FeaturizerRegistry)


def test_cell_line_featurizer_registry_uses_fixed_identity() -> None:
    registry = CellLineFeaturizerRegistry()

    assert registry._registry_id == "cell_line_featurizer"
    assert registry._label == "Cell line featurizer"
    assert registry._display_name == "cell_line_featurizers"


def test_cell_line_featurizer_registry_declares_the_cell_line_side() -> None:
    assert CellLineFeaturizerRegistry()._side == "cell_line"


def test_module_singleton_is_a_cell_line_featurizer_registry() -> None:
    assert isinstance(cell_line_featurizer_registry, CellLineFeaturizerRegistry)


def test_isolated_registry_registers_with_a_contract() -> None:
    registry = CellLineFeaturizerRegistry()

    @registry.register("localCellLine", description="local", contract=FeatureFormat.NUMERIC_MATRIX)
    class LocalCellLine:
        pass

    assert registry.get("localCellLine") is LocalCellLine
    assert vars(LocalCellLine)["contract"] == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def test_registration_stamps_the_cell_line_side_onto_the_class() -> None:
    registry = CellLineFeaturizerRegistry()

    @registry.register("sidedCellLine", description="sided", contract=FeatureFormat.NUMERIC_MATRIX)
    class SidedCellLine:
        pass

    assert SidedCellLine.side == "cell_line"


def test_registration_defaults_the_storage_key_to_the_registry_name() -> None:
    registry = CellLineFeaturizerRegistry()

    @registry.register("storedCellLine", description="stored", contract=FeatureFormat.NUMERIC_MATRIX)
    class StoredCellLine:
        pass

    assert StoredCellLine.storage_key == "storedCellLine"


def test_metadata_reports_the_cell_line_display_name() -> None:
    registry = CellLineFeaturizerRegistry()

    @registry.register("metaCellLine", description="meta", contract=FeatureFormat.GRAPH)
    class MetaCellLine:
        pass

    assert registry.get_metadata("metaCellLine")["registry"] == "cell_line_featurizers"


def test_metadata_listing_covers_every_registered_featurizer() -> None:
    registry = _registry_with_two_featurizers()

    assert {row["name"] for row in registry.list_metadata()} == {"coreFeat", "baselineFeat"}


def test_metadata_listing_filters_by_tag() -> None:
    registry = _registry_with_two_featurizers()

    assert {row["name"] for row in registry.list_metadata(tag="baseline")} == {"baselineFeat"}


def _registry_with_two_featurizers() -> CellLineFeaturizerRegistry:
    registry = CellLineFeaturizerRegistry()

    @registry.register("coreFeat", description="core", contract=FeatureFormat.NUMERIC_MATRIX)
    class CoreFeat:
        pass

    @registry.register(
        "baselineFeat",
        description="baseline",
        tags=("baseline",),
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class BaselineFeat:
        pass

    return registry
