"""Tests for DrugFeaturizerRegistry type and singleton."""

from __future__ import annotations

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from drevalpy.registry.drug_featurizer._registry import DrugFeaturizerRegistry
from drevalpy.registry.featurizer import FeaturizerRegistry


def test_drug_featurizer_registry_subclasses_the_shared_base() -> None:
    assert issubclass(DrugFeaturizerRegistry, FeaturizerRegistry)


def test_drug_featurizer_registry_uses_fixed_identity() -> None:
    registry = DrugFeaturizerRegistry()

    assert registry._registry_id == "drug_featurizer"
    assert registry._label == "Drug featurizer"
    assert registry._display_name == "drug_featurizers"


def test_drug_featurizer_registry_declares_the_drug_side() -> None:
    assert DrugFeaturizerRegistry()._side == "drug"


def test_module_singleton_is_a_drug_featurizer_registry() -> None:
    assert isinstance(drug_featurizer_registry, DrugFeaturizerRegistry)


def test_isolated_registry_registers_with_a_contract() -> None:
    registry = DrugFeaturizerRegistry()

    @registry.register("localDrug", description="local", contract=FeatureFormat.GRAPH)
    class LocalDrug:
        pass

    assert registry.get("localDrug") is LocalDrug
    assert vars(LocalDrug)["contract"] == FeatureContract(format=FeatureFormat.GRAPH)


def test_registration_stamps_the_drug_side_onto_the_class() -> None:
    registry = DrugFeaturizerRegistry()

    @registry.register("sidedDrug", description="sided", contract=FeatureFormat.NUMERIC_MATRIX)
    class SidedDrug:
        pass

    assert SidedDrug.side == "drug"


def test_registration_keeps_an_explicit_storage_key() -> None:
    registry = DrugFeaturizerRegistry()

    @registry.register("storedDrug", description="stored", contract=FeatureFormat.NUMERIC_MATRIX)
    class StoredDrug:
        storage_key = "shared_bucket"

    assert StoredDrug.storage_key == "shared_bucket"


def test_metadata_reports_the_drug_display_name() -> None:
    registry = DrugFeaturizerRegistry()

    @registry.register("metaDrug", description="meta", contract=FeatureFormat.GRAPH)
    class MetaDrug:
        pass

    assert registry.get_metadata("metaDrug")["registry"] == "drug_featurizers"


def test_metadata_listing_covers_every_registered_featurizer() -> None:
    registry = _registry_with_two_featurizers()

    assert {row["name"] for row in registry.list_metadata()} == {"coreDrug", "baselineDrug"}


def test_metadata_listing_filters_by_tag() -> None:
    registry = _registry_with_two_featurizers()

    assert {row["name"] for row in registry.list_metadata(tag="baseline")} == {"baselineDrug"}


def test_metadata_listing_carries_the_registered_tags() -> None:
    registry = _registry_with_two_featurizers()

    assert registry.list_metadata(tag="baseline")[0]["tags"] == frozenset({"baseline"})


def _registry_with_two_featurizers() -> DrugFeaturizerRegistry:
    registry = DrugFeaturizerRegistry()

    @registry.register("coreDrug", description="core", contract=FeatureFormat.NUMERIC_MATRIX)
    class CoreDrug:
        pass

    @registry.register(
        "baselineDrug",
        description="baseline",
        tags=("baseline",),
        contract=FeatureFormat.GRAPH,
    )
    class BaselineDrug:
        pass

    return registry
