"""Tests for public drug featurizer registration and lookup helpers."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.registry.drug_featurizer import (
    get as get_drug_featurizer,
)
from drevalpy.registry.drug_featurizer import (
    list as list_drug_featurizers,
)
from drevalpy.registry.drug_featurizer import (
    metadata as get_drug_featurizer_metadata,
)
from drevalpy.registry.drug_featurizer import (
    register as register_drug_featurizer,
)
from tests.registry._helpers import isolated_component_registries


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    yield from isolated_component_registries()


def test_register_and_lookup_drug_featurizer() -> None:
    @register_drug_featurizer(
        "dummyDrug",
        description="test drug",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class DummyDrug:
        pass

    assert get_drug_featurizer("dummyDrug") is DummyDrug
    assert vars(DummyDrug)["contract"] == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    assert "dummyDrug" in list_drug_featurizers()


def test_unknown_drug_component_fails() -> None:
    with pytest.raises(ValueError, match="Unknown Drug featurizer"):
        get_drug_featurizer("missing")


def test_get_drug_metadata_includes_output_format() -> None:
    @register_drug_featurizer(
        "graphDrug",
        description="graph",
        contract=FeatureFormat.GRAPH,
    )
    class GraphDrug:
        pass

    meta = get_drug_featurizer_metadata("graphDrug")
    assert meta["output_format"] == "graph"
    assert meta["description"] == "graph"


def test_duplicate_drug_class_and_decorator_contract_fails() -> None:
    with pytest.raises(ValueError, match="do not set contract on the class body"):

        @register_drug_featurizer(
            "drugConflict",
            description="conflict",
            contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class DrugConflict:
            contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def test_featurizer_registration_requires_explicit_contract() -> None:
    with pytest.raises(TypeError, match="contract"):
        register_drug_featurizer("noContractDrug", description="missing contract")  # type: ignore[call-arg]
