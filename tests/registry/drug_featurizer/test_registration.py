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


def test_the_decorator_contract_overrides_the_class_body() -> None:
    @register_drug_featurizer(
        "drugOverridden",
        description="decorator wins",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class DrugOverridden:
        contract = FeatureContract(format=FeatureFormat.GRAPH)

    assert vars(DrugOverridden)["contract"] == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def test_a_class_body_contract_is_a_valid_declaration() -> None:
    @register_drug_featurizer("drugBodyContract", description="declares on the class body")
    class DrugBodyContract:
        contract = FeatureContract(format=FeatureFormat.GRAPH)

    assert get_drug_featurizer_metadata("drugBodyContract")["output_format"] == "graph"


def test_a_featurizer_declaring_no_contract_anywhere_is_rejected() -> None:
    with pytest.raises(ValueError, match="no contract declared"):

        @register_drug_featurizer("noContractDrug", description="missing contract")
        class NoContractDrug:
            pass


def test_an_invalid_class_body_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="class-body contract is invalid"):

        @register_drug_featurizer("badContractDrug", description="wrong type")
        class BadContractDrug:
            contract = "numeric_matrix_but_a_plain_string"
