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
def test_side_registry_declares_its_side_and_identity(side, registry_cls, singleton) -> None:
    registry = registry_cls()

    assert issubclass(registry_cls, FeaturizerRegistry)
    assert isinstance(singleton, registry_cls)
    assert registry._side == side
    assert (registry._registry_id, registry._label, registry._display_name) == (
        f"{side}_featurizer",
        f"{side.capitalize()} featurizer",
        f"{side}_featurizers",
    )


@SIDES
def test_registration_stamps_the_side_and_keeps_an_explicit_storage_key(side, registry_cls, singleton) -> None:
    registry = registry_cls()

    @registry.register("sided", description="sided", contract=FeatureFormat.NUMERIC_MATRIX)
    class Sided:
        storage_key = "shared_bucket"

    assert Sided.side == side
    assert Sided.storage_key == "shared_bucket"


@SIDES
def test_metadata_reports_the_side_and_listing_filters_by_tag(side, registry_cls, singleton) -> None:
    registry = registry_cls()

    @registry.register("core", description="core", contract=FeatureFormat.NUMERIC_MATRIX)
    class Core:
        pass

    @registry.register("baseline", description="baseline", tags=("baseline",), contract=FeatureFormat.GRAPH)
    class Baseline:
        pass

    assert registry.get_metadata("core")["registry"] == f"{side}_featurizers"
    assert {row["name"] for row in registry.list_metadata()} == {"core", "baseline"}
    tagged = registry.list_metadata(tag="baseline")
    assert [row["name"] for row in tagged] == ["baseline"]
    assert tagged[0]["tags"] == frozenset({"baseline"})


def test_isolated_registry_registers_with_contract_and_restores_after_clear() -> None:
    registry = FeaturizerRegistry("test", "Test featurizer", "test_featurizers")

    @registry.register("localFeat", description="local", contract=FeatureFormat.GRAPH)
    class LocalFeat:
        pass

    assert registry.get("localFeat") is LocalFeat
    assert vars(LocalFeat)["contract"] == FeatureContract(format=FeatureFormat.GRAPH)
    assert vars(LocalFeat)["registry_name"] == "localFeat"
    assert not hasattr(LocalFeat, "side")

    registry.clear()
    assert registry.list_names() == []

    registry.register_existing("localFeat", LocalFeat)
    assert registry.get("localFeat") is LocalFeat


@pytest.mark.parametrize("side", ["sample", "drug"])
def test_get_featurizer_resolves_a_registered_name_and_rejects_an_unknown_one(side: str) -> None:
    registry = {"sample": sample_featurizer_registry, "drug": drug_featurizer_registry}[side]
    name = registry.list_names()[0]

    assert get_featurizer(side, name) is registry.get(name)  # type: ignore[arg-type]
    with pytest.raises(UnknownNameError, match="Unknown .*featurizer: 'nope'"):
        get_featurizer(side, "nope")  # type: ignore[arg-type]


def test_get_featurizer_rejects_an_unknown_side() -> None:
    with pytest.raises(UnknownNameError, match=r"Unknown featurizer side 'protein'.*\['sample', 'drug'\]"):
        get_featurizer("protein", "identity")  # type: ignore[arg-type]
