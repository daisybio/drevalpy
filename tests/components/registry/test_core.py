"""Tests for internal component registries."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.registry import (
    clear_cell_line_featurizer_registry,
    clear_drug_featurizer_registry,
    clear_predictor_registry,
    get_cell_line_featurizer,
    get_cell_line_featurizer_metadata,
    get_drug_featurizer,
    get_drug_featurizer_metadata,
    list_cell_line_featurizer_metadata,
    list_cell_line_featurizers,
    list_drug_featurizer_metadata,
    list_drug_featurizers,
    register_cell_line_featurizer,
    register_drug_featurizer,
    register_predictor,
)
from drevalpy.components.registry.core import Registry


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    clear_cell_line_featurizer_registry()
    clear_drug_featurizer_registry()
    clear_predictor_registry()
    yield
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()


def test_register_and_lookup_cell_line_featurizer() -> None:
    @register_cell_line_featurizer(
        "dummyCellLine",
        description="test cell line",
        category="native",
        contract=FeatureKind.DENSE,
    )
    class DummyCellLine:
        pass

    assert get_cell_line_featurizer("dummyCellLine") is DummyCellLine
    assert vars(DummyCellLine)["contract"] == FeatureContract(kind=FeatureKind.DENSE)
    assert "dummyCellLine" in list_cell_line_featurizers()


def test_register_and_lookup_drug_featurizer() -> None:
    @register_drug_featurizer(
        "dummyDrug",
        description="test drug",
        category="native",
        contract=FeatureKind.DENSE,
    )
    class DummyDrug:
        pass

    assert get_drug_featurizer("dummyDrug") is DummyDrug
    assert vars(DummyDrug)["contract"] == FeatureContract(kind=FeatureKind.DENSE)
    assert "dummyDrug" in list_drug_featurizers()


def test_duplicate_registration_fails() -> None:
    @register_cell_line_featurizer(
        "dup",
        description="first",
        category="native",
        contract=FeatureKind.DENSE,
    )
    class First:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @register_cell_line_featurizer(
            "dup",
            description="second",
            category="native",
            contract=FeatureKind.DENSE,
        )
        class Second:
            pass


def test_unknown_drug_component_fails() -> None:
    with pytest.raises(ValueError, match="Unknown Drug featurizer"):
        get_drug_featurizer("missing")


def test_unknown_component_fails() -> None:
    with pytest.raises(ValueError, match="Unknown Cell line featurizer"):
        get_cell_line_featurizer("missing")


def test_drug_metadata_listing_and_category_filter() -> None:
    @register_drug_featurizer(
        "nativeDrug",
        description="native",
        category="native",
        contract=FeatureKind.DENSE,
    )
    class NativeDrug:
        pass

    @register_drug_featurizer(
        "baselineDrug",
        description="baseline",
        category="baseline",
        contract=FeatureKind.GRAPH,
    )
    class BaselineDrug:
        pass

    all_rows = list_drug_featurizer_metadata()
    assert len(all_rows) == 2
    baseline_rows = list_drug_featurizer_metadata(category="baseline")
    assert len(baseline_rows) == 1
    assert baseline_rows[0]["name"] == "baselineDrug"


def test_metadata_listing_and_category_filter() -> None:
    @register_cell_line_featurizer(
        "nativeEnc",
        description="native",
        category="native",
        contract=FeatureKind.DENSE,
    )
    class NativeEnc:
        pass

    @register_cell_line_featurizer(
        "baselineEnc",
        description="baseline",
        category="baseline",
        contract=FeatureKind.DENSE,
    )
    class BaselineEnc:
        pass

    all_rows = list_cell_line_featurizer_metadata()
    assert len(all_rows) == 2
    baseline_rows = list_cell_line_featurizer_metadata(category="baseline")
    assert len(baseline_rows) == 1
    assert baseline_rows[0]["name"] == "baselineEnc"


def test_get_drug_metadata_includes_output_type() -> None:
    @register_drug_featurizer(
        "graphDrug",
        description="graph",
        category="native",
        contract=FeatureKind.GRAPH,
    )
    class GraphDrug:
        pass

    meta = get_drug_featurizer_metadata("graphDrug")
    assert meta["output_type"] == "graph"
    assert meta["description"] == "graph"


def test_get_metadata_includes_output_type() -> None:
    @register_cell_line_featurizer(
        "graphEnc",
        description="graph",
        category="native",
        contract=FeatureKind.GRAPH,
    )
    class GraphEnc:
        pass

    meta = get_cell_line_featurizer_metadata("graphEnc")
    assert meta["output_type"] == "graph"
    assert meta["description"] == "graph"


def test_decorator_returns_original_class() -> None:
    @register_predictor(
        "dummyPred",
        description="pred",
        category="baseline",
        cell_line_contract=FeatureKind.DENSE,
        drug_contract=FeatureKind.DENSE,
    )
    class DummyPred:
        requires_drug_featurizer = False
        supported_modes = {"regression"}

    assert vars(DummyPred)["registry_name"] == "dummyPred"
    assert vars(DummyPred)["cell_line_contract"] == FeatureContract(kind=FeatureKind.DENSE)
    assert vars(DummyPred)["drug_contract"] == FeatureContract(kind=FeatureKind.DENSE)


def test_duplicate_drug_class_and_decorator_contract_fails() -> None:
    with pytest.raises(ValueError, match="already defines a featurizer contract"):

        @register_drug_featurizer(
            "drugConflict",
            description="conflict",
            category="native",
            contract=FeatureKind.DENSE,
        )
        class DrugConflict:
            contract = FeatureContract(kind=FeatureKind.DENSE)


def test_duplicate_class_and_decorator_contract_fails() -> None:
    with pytest.raises(ValueError, match="already defines a featurizer contract"):

        @register_cell_line_featurizer(
            "conflict",
            description="conflict",
            category="native",
            contract=FeatureKind.DENSE,
        )
        class Conflict:
            contract = FeatureContract(kind=FeatureKind.DENSE)


def test_featurizer_registration_requires_explicit_contract() -> None:
    with pytest.raises(ValueError, match="missing=\\['contract'\\]"):

        @register_drug_featurizer(
            "noContractDrug",
            description="missing contract",
            category="native",
        )
        class NoContractDrug:
            pass


def test_registry_clear() -> None:
    registry = Registry("test", "Test component", "test_components", lambda *_: {})
    decorated = registry.register("x", description="x", category="native")

    @decorated
    class X:
        pass

    assert registry.list_names() == ["x"]
    registry.clear()
    assert registry.list_names() == []
