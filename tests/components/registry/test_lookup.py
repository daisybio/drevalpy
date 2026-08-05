"""Tests for public registry lookup and registration decorators."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts import FeatureContract, FeatureFormat
from drevalpy.components.predictors.feature_free import FeatureFreePredictor
from drevalpy.components.registry import (
    get_cell_line_featurizer,
    get_cell_line_featurizer_metadata,
    get_drug_featurizer,
    get_drug_featurizer_metadata,
    get_predictor_metadata,
    list_cell_line_featurizer_metadata,
    list_cell_line_featurizers,
    list_drug_featurizer_metadata,
    list_drug_featurizers,
    register_cell_line_featurizer,
    register_drug_featurizer,
    register_predictor,
)
from drevalpy.components.registry.featurizer import (
    FeaturizerRegistry,
    cell_line_featurizer_registry,
    drug_featurizer_registry,
)
from drevalpy.components.registry.predictor import predictor_registry


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    cell_line_featurizer_registry.clear()
    drug_featurizer_registry.clear()
    predictor_registry.clear()
    yield
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()


def test_register_and_lookup_cell_line_featurizer() -> None:
    @register_cell_line_featurizer(
        "dummyCellLine",
        description="test cell line",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class DummyCellLine:
        pass

    assert get_cell_line_featurizer("dummyCellLine") is DummyCellLine
    assert vars(DummyCellLine)["contract"] == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    assert "dummyCellLine" in list_cell_line_featurizers()


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


def test_duplicate_registration_fails() -> None:
    @register_cell_line_featurizer(
        "dup",
        description="first",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class First:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @register_cell_line_featurizer(
            "dup",
            description="second",
            contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class Second:
            pass


def test_unknown_drug_component_fails() -> None:
    with pytest.raises(ValueError, match="Unknown Drug featurizer"):
        get_drug_featurizer("missing")


def test_unknown_component_fails() -> None:
    with pytest.raises(ValueError, match="Unknown Cell line featurizer"):
        get_cell_line_featurizer("missing")


def test_drug_metadata_listing_and_tag_filter() -> None:
    @register_drug_featurizer(
        "coreDrug",
        description="core",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class CoreDrug:
        pass

    @register_drug_featurizer(
        "baselineDrug",
        description="baseline",
        tags=("baseline",),
        contract=FeatureFormat.GRAPH,
    )
    class BaselineDrug:
        pass

    all_rows = list_drug_featurizer_metadata()
    assert {row["name"] for row in all_rows if row["name"] in {"coreDrug", "baselineDrug"}} == {
        "coreDrug",
        "baselineDrug",
    }
    baseline_rows = list_drug_featurizer_metadata(tag="baseline")
    assert {row["name"] for row in baseline_rows} == {"baselineDrug"}
    assert baseline_rows[0]["tags"] == frozenset({"baseline"})


def test_metadata_listing_and_tag_filter() -> None:
    @register_cell_line_featurizer(
        "coreFeat",
        description="core",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class CoreFeat:
        pass

    @register_cell_line_featurizer(
        "baselineFeat",
        description="baseline",
        tags=("baseline",),
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class BaselineFeat:
        pass

    all_rows = list_cell_line_featurizer_metadata()
    assert {row["name"] for row in all_rows if row["name"] in {"coreFeat", "baselineFeat"}} == {
        "coreFeat",
        "baselineFeat",
    }
    baseline_rows = list_cell_line_featurizer_metadata(tag="baseline")
    assert {row["name"] for row in baseline_rows} == {"baselineFeat"}


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


def test_get_metadata_includes_output_format() -> None:
    @register_cell_line_featurizer(
        "graphFeat",
        description="graph",
        contract=FeatureFormat.GRAPH,
    )
    class GraphFeat:
        pass

    meta = get_cell_line_featurizer_metadata("graphFeat")
    assert meta["output_format"] == "graph"
    assert meta["description"] == "graph"
    assert meta["tags"] == frozenset()


def test_decorator_returns_original_class() -> None:
    @register_predictor(
        "dummyPred",
        description="pred",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class DummyPred(FeatureFreePredictor):
        pass

    assert vars(DummyPred)["registry_name"] == "dummyPred"
    assert vars(DummyPred)["cell_line_contract"] == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    assert vars(DummyPred)["drug_contract"] == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def test_predictor_metadata_catalog_shape() -> None:
    @register_predictor(
        "catalogPred",
        description="catalog shape",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class CatalogPred(FeatureFreePredictor):
        pass

    meta = get_predictor_metadata("catalogPred")
    assert meta["input_interface"] == "feature_free"
    assert meta["description"] == "catalog shape"
    assert meta["tags"] == frozenset()
    for dropped in (
        "cell_line_format",
        "drug_format",
        "supported_modes",
        "supported_scopes",
        "supports_early_stopping",
        "requires_drug_featurizer",
        "required_cell_line_views",
        "required_drug_views",
    ):
        assert dropped not in meta


def test_duplicate_drug_class_and_decorator_contract_fails() -> None:
    with pytest.raises(ValueError, match="do not set contract on the class body"):

        @register_drug_featurizer(
            "drugConflict",
            description="conflict",
            contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class DrugConflict:
            contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def test_duplicate_class_and_decorator_contract_fails() -> None:
    with pytest.raises(ValueError, match="do not set contract on the class body"):

        @register_cell_line_featurizer(
            "conflict",
            description="conflict",
            contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class Conflict:
            contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def test_duplicate_predictor_class_and_decorator_contract_fails() -> None:
    with pytest.raises(ValueError, match="do not set cell_line_contract on the class body"):

        @register_predictor(
            "predConflict",
            description="conflict",
            cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
            drug_contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class PredConflict:
            cell_line_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def test_duplicate_predictor_drug_class_and_decorator_contract_fails() -> None:
    with pytest.raises(ValueError, match="do not set drug_contract on the class body"):

        @register_predictor(
            "predDrugConflict",
            description="conflict",
            cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
            drug_contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class PredDrugConflict:
            drug_contract = FeatureContract(format=FeatureFormat.GRAPH)


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


def test_featurizer_registration_requires_explicit_contract() -> None:
    with pytest.raises(TypeError, match="contract"):
        register_drug_featurizer("noContractDrug", description="missing contract")  # type: ignore[call-arg]


def test_predictor_registration_requires_explicit_contracts() -> None:
    with pytest.raises(TypeError, match="contract"):
        register_predictor("noContractPred", description="missing contracts")  # type: ignore[call-arg]


def test_registry_clear() -> None:
    registry = FeaturizerRegistry("test", "Test component", "test_components")
    decorated = registry.register("x", description="x", contract=FeatureFormat.NUMERIC_MATRIX)

    @decorated
    class X:
        pass

    assert registry.list_names() == ["x"]
    registry.clear()
    assert registry.list_names() == []
