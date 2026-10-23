"""Tests for FeaturizerRegistry, its per-side subclasses and the side singletons."""

from __future__ import annotations

import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.errors import UnknownNameError
from drevalpy.registry.featurizer import (
    DrugFeaturizerRegistry,
    FeaturizerRegistry,
    SampleFeaturizerRegistry,
    drug_featurizer_registry,
    get_featurizer,
    sample_featurizer_registry,
)

SIDES = pytest.mark.parametrize(
    ("side", "registry_cls", "singleton"),
    [
        ("sample", SampleFeaturizerRegistry, sample_featurizer_registry),
        ("drug", DrugFeaturizerRegistry, drug_featurizer_registry),
    ],
)


@SIDES
def test_side_registry_subclasses_the_shared_base_and_declares_its_side(side, registry_cls, singleton) -> None:
    assert issubclass(registry_cls, FeaturizerRegistry)
    assert registry_cls()._side == side
    assert isinstance(singleton, registry_cls)


@SIDES
def test_side_registry_uses_fixed_identity(side, registry_cls, singleton) -> None:
    registry = registry_cls()

    assert registry._registry_id == f"{side}_featurizer"
    assert registry._label == f"{side.capitalize()} featurizer"
    assert registry._display_name == f"{side}_featurizers"
    assert singleton._registry_id == registry._registry_id


@SIDES
def test_registration_stamps_the_side_onto_the_class(side, registry_cls, singleton) -> None:
    registry = registry_cls()

    @registry.register("sided", description="sided", contract=FeatureFormat.NUMERIC_MATRIX)
    class Sided:
        pass

    assert Sided.side == side


@SIDES
def test_registration_keeps_an_explicit_storage_key(side, registry_cls, singleton) -> None:
    registry = registry_cls()

    @registry.register("stored", description="stored", contract=FeatureFormat.NUMERIC_MATRIX)
    class Stored:
        storage_key = "shared_bucket"

    assert Stored.storage_key == "shared_bucket"


@SIDES
def test_metadata_reports_the_side_display_name(side, registry_cls, singleton) -> None:
    registry = registry_cls()

    @registry.register("meta", description="meta", contract=FeatureFormat.GRAPH)
    class Meta:
        pass

    assert registry.get_metadata("meta")["registry"] == f"{side}_featurizers"


@SIDES
def test_metadata_listing_covers_and_filters_by_tag(side, registry_cls, singleton) -> None:
    registry = registry_cls()

    @registry.register("core", description="core", contract=FeatureFormat.NUMERIC_MATRIX)
    class Core:
        pass

    @registry.register("baseline", description="baseline", tags=("baseline",), contract=FeatureFormat.GRAPH)
    class Baseline:
        pass

    assert {row["name"] for row in registry.list_metadata()} == {"core", "baseline"}
    tagged = registry.list_metadata(tag="baseline")
    assert [row["name"] for row in tagged] == ["baseline"]
    assert tagged[0]["tags"] == frozenset({"baseline"})


def test_isolated_featurizer_registry_registers_with_contract() -> None:
    registry = FeaturizerRegistry("test", "Test featurizer", "test_featurizers")

    @registry.register("localFeat", description="local", contract=FeatureFormat.GRAPH)
    class LocalFeat:
        pass

    assert registry.get("localFeat") is LocalFeat
    assert vars(LocalFeat)["contract"] == FeatureContract(format=FeatureFormat.GRAPH)
    assert not hasattr(LocalFeat, "side")


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


@pytest.mark.parametrize("side", ["sample", "drug"])
def test_get_featurizer_resolves_a_registered_name_on_each_side(side: str) -> None:
    registry = {"sample": sample_featurizer_registry, "drug": drug_featurizer_registry}[side]
    name = registry.list_names()[0]

    assert get_featurizer(side, name) is registry.get(name)  # type: ignore[arg-type]


def test_get_featurizer_rejects_an_unknown_side() -> None:
    with pytest.raises(UnknownNameError, match=r"Unknown featurizer side 'protein'.*\['sample', 'drug'\]"):
        get_featurizer("protein", "identity")  # type: ignore[arg-type]


@pytest.mark.parametrize("side", ["sample", "drug"])
def test_get_featurizer_rejects_an_unknown_name(side: str) -> None:
    with pytest.raises(UnknownNameError, match="Unknown .*featurizer: 'nope'"):
        get_featurizer(side, "nope")  # type: ignore[arg-type]
