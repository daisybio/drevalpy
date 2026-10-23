"""Tests for the side packages that bind :mod:`drevalpy.registry.featurizer` to one entity side.

``drevalpy.registry.sample_featurizer`` and ``drevalpy.registry.drug_featurizer`` hold no logic of their
own: each exposes the ``register`` / ``get`` / ``list`` / ``table`` / ``metadata`` helpers of its side's
registry. So every test runs against both packages.
"""

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


@SIDES
def test_duplicate_registration_fails(package: ModuleType) -> None:
    @package.register("dup", description="first", contract=FeatureFormat.NUMERIC_MATRIX)
    class First:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @package.register("dup", description="second", contract=FeatureFormat.NUMERIC_MATRIX)
        class Second:
            pass


@SIDES
def test_unknown_name_fails(package: ModuleType) -> None:
    with pytest.raises(UnknownNameError, match=r"Unknown (Sample|Drug) featurizer: 'missing'"):
        package.get("missing")


@SIDES
def test_metadata_includes_output_format_and_description(package: ModuleType) -> None:
    @package.register("graphFeat", description="graph", contract=FeatureFormat.GRAPH)
    class GraphFeat:
        pass

    meta = package.metadata("graphFeat")

    assert meta["output_format"] == "graph"
    assert meta["description"] == "graph"
    assert meta["tags"] == frozenset()


@SIDES
def test_the_decorator_contract_overrides_the_class_body(package: ModuleType) -> None:
    @package.register("overridden", description="decorator wins", contract=FeatureFormat.NUMERIC_MATRIX)
    class Overridden:
        contract = FeatureContract(format=FeatureFormat.GRAPH)

    assert vars(Overridden)["contract"] == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


@SIDES
def test_a_class_body_contract_is_a_valid_declaration(package: ModuleType) -> None:
    @package.register("bodyContract", description="declares on the class body")
    class BodyContract:
        contract = FeatureContract(format=FeatureFormat.GRAPH)

    assert package.metadata("bodyContract")["output_format"] == "graph"


@SIDES
def test_a_class_body_format_shorthand_is_normalized(package: ModuleType) -> None:
    @package.register("shorthandContract", description="format shorthand")
    class ShorthandContract:
        contract = FeatureFormat.RAGGED_SEQUENCE

    assert vars(ShorthandContract)["contract"] == FeatureContract(format=FeatureFormat.RAGGED_SEQUENCE)


@SIDES
def test_a_featurizer_declaring_no_contract_anywhere_is_rejected(package: ModuleType) -> None:
    with pytest.raises(ValueError, match="no contract declared"):

        @package.register("noContract", description="missing contract")
        class NoContract:
            pass


@SIDES
def test_an_invalid_class_body_contract_is_rejected(package: ModuleType) -> None:
    with pytest.raises(ValueError, match="class-body contract is invalid"):

        @package.register("badContract", description="wrong type")
        class BadContract:
            contract = "numeric_matrix_but_a_plain_string"
