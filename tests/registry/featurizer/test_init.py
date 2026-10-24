"""Tests for the side packages that bind :mod:`drevalpy.registry.featurizer` to one entity side (run against both)."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.errors import UnknownNameError
from drevalpy.registry import drug_featurizer, sample_featurizer
from drevalpy.registry.featurizer import (
    DrugFeaturizerRegistry,
    SampleFeaturizerRegistry,
    drug_featurizer_registry,
    sample_featurizer_registry,
)
from tests.registry._helpers import isolated_component_registries

SIDES = pytest.mark.parametrize("package", [sample_featurizer, drug_featurizer], ids=["sample", "drug"])


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    yield from isolated_component_registries()


def test_the_side_packages_expose_the_registries_of_the_featurizer_package() -> None:
    assert sample_featurizer.sample_featurizer_registry is sample_featurizer_registry
    assert sample_featurizer.SampleFeaturizerRegistry is SampleFeaturizerRegistry
    assert drug_featurizer.drug_featurizer_registry is drug_featurizer_registry
    assert drug_featurizer.DrugFeaturizerRegistry is DrugFeaturizerRegistry


@SIDES
def test_all_names_the_registry_and_the_five_helpers(package: ModuleType) -> None:
    helpers = {"get", "list", "metadata", "register", "table"}

    assert helpers <= set(package.__all__)
    assert all(callable(getattr(package, name)) for name in helpers)


@SIDES
def test_register_and_lookup(package: ModuleType) -> None:
    @package.register("dummy", description="test", contract=FeatureFormat.NUMERIC_MATRIX)
    class Dummy:
        pass

    assert package.get("dummy") is Dummy
    assert vars(Dummy)["contract"] == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    assert "dummy" in package.list()
    assert "dummy" in package.table()["Name"].tolist()
    meta = package.metadata("dummy")
    assert (meta["output_format"], meta["description"], meta["tags"]) == ("numeric_matrix", "test", frozenset())


@SIDES
def test_duplicate_and_unknown_names_fail(package: ModuleType) -> None:
    @package.register("dup", description="first", contract=FeatureFormat.NUMERIC_MATRIX)
    class First:
        pass

    with pytest.raises(ValueError, match="already registered"):
        package.register("dup", description="second", contract=FeatureFormat.NUMERIC_MATRIX)(type("Second", (), {}))
    with pytest.raises(UnknownNameError, match=r"Unknown (Sample|Drug) featurizer: 'missing'"):
        package.get("missing")


@SIDES
def test_the_decorator_contract_overrides_the_class_body(package: ModuleType) -> None:
    @package.register("overridden", description="decorator wins", contract=FeatureFormat.NUMERIC_MATRIX)
    class Overridden:
        contract = FeatureContract(format=FeatureFormat.GRAPH)

    assert vars(Overridden)["contract"] == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


@SIDES
def test_a_class_body_contract_is_a_valid_declaration_and_a_format_shorthand_is_normalized(
    package: ModuleType,
) -> None:
    @package.register("bodyContract", description="declares on the class body")
    class BodyContract:
        contract = FeatureContract(format=FeatureFormat.GRAPH)

    @package.register("shorthandContract", description="format shorthand")
    class ShorthandContract:
        contract = FeatureFormat.RAGGED_SEQUENCE

    assert package.metadata("bodyContract")["output_format"] == "graph"
    assert vars(ShorthandContract)["contract"] == FeatureContract(format=FeatureFormat.RAGGED_SEQUENCE)


@SIDES
@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({}, "no contract declared"),
        ({"contract": "numeric_matrix_but_a_plain_string"}, "class-body contract is invalid"),
    ],
    ids=["missing", "wrong-type"],
)
def test_a_missing_or_invalid_contract_is_rejected(package: ModuleType, body: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        package.register("badContract", description="bad")(type("BadContract", (), body))
