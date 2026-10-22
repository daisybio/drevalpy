"""Tests for SampleFeaturizerRegistry type and singleton."""

from __future__ import annotations

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.registry.featurizer import FeaturizerRegistry
from drevalpy.registry.sample_featurizer import sample_featurizer_registry
from drevalpy.registry.sample_featurizer._registry import SampleFeaturizerRegistry


def test_sample_featurizer_registry_subclasses_the_shared_base() -> None:
    assert issubclass(SampleFeaturizerRegistry, FeaturizerRegistry)


def test_sample_featurizer_registry_uses_fixed_identity() -> None:
    registry = SampleFeaturizerRegistry()

    assert registry._registry_id == "sample_featurizer"
    assert registry._label == "Sample featurizer"
    assert registry._display_name == "sample_featurizers"


def test_sample_featurizer_registry_declares_the_sample_side() -> None:
    assert SampleFeaturizerRegistry()._side == "sample"


def test_module_singleton_is_a_sample_featurizer_registry() -> None:
    assert isinstance(sample_featurizer_registry, SampleFeaturizerRegistry)


def test_isolated_registry_registers_with_a_contract() -> None:
    registry = SampleFeaturizerRegistry()

    @registry.register("localSample", description="local", contract=FeatureFormat.NUMERIC_MATRIX)
    class LocalSample:
        pass

    assert registry.get("localSample") is LocalSample
    assert vars(LocalSample)["contract"] == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def test_registration_stamps_the_sample_side_onto_the_class() -> None:
    registry = SampleFeaturizerRegistry()

    @registry.register("sidedSample", description="sided", contract=FeatureFormat.NUMERIC_MATRIX)
    class SidedSample:
        pass

    assert SidedSample.side == "sample"


def test_registration_defaults_the_storage_key_to_the_registry_name() -> None:
    registry = SampleFeaturizerRegistry()

    @registry.register("storedSample", description="stored", contract=FeatureFormat.NUMERIC_MATRIX)
    class StoredSample:
        pass

    assert StoredSample.storage_key == "storedSample"


def test_metadata_reports_the_sample_display_name() -> None:
    registry = SampleFeaturizerRegistry()

    @registry.register("metaSample", description="meta", contract=FeatureFormat.GRAPH)
    class MetaSample:
        pass

    assert registry.get_metadata("metaSample")["registry"] == "sample_featurizers"


def test_metadata_listing_covers_every_registered_featurizer() -> None:
    registry = _registry_with_two_featurizers()

    assert {row["name"] for row in registry.list_metadata()} == {"coreFeat", "baselineFeat"}


def test_metadata_listing_filters_by_tag() -> None:
    registry = _registry_with_two_featurizers()

    assert {row["name"] for row in registry.list_metadata(tag="baseline")} == {"baselineFeat"}


def _registry_with_two_featurizers() -> SampleFeaturizerRegistry:
    registry = SampleFeaturizerRegistry()

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
