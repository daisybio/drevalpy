"""Tests for public featurizer registration and lookup helpers."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.core.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.registry import (
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
)
from drevalpy.components.registry.featurizer_registry import (
    cell_line_featurizer_registry,
    drug_featurizer_registry,
)
from drevalpy.components.registry.predictor_registry import predictor_registry


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    cell_line_featurizer_registry.clear()
    drug_featurizer_registry.clear()
    predictor_registry.clear()
    yield
    from drevalpy.components.core.plugins.register_builtins import register_builtin_components

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


def test_featurizer_registration_requires_explicit_contract() -> None:
    with pytest.raises(TypeError, match="contract"):
        register_drug_featurizer("noContractDrug", description="missing contract")  # type: ignore[call-arg]


def test_featurizer_registration_requires_declared_input_views() -> None:
    from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer

    with pytest.raises(ValueError, match="does not declare its input views"):

        @register_cell_line_featurizer(
            "undeclaredViews",
            description="no input views declared",
            contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class UndeclaredViews(CellLineFeaturizer):
            def fit(self, features, *, entity_ids=None, context=None):
                return self

            def transform(self, features, entity_ids):
                raise NotImplementedError

            @property
            def output_dim(self) -> int:
                return 0


def test_declared_input_views_allow_registration() -> None:
    from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer

    @register_cell_line_featurizer(
        "declaredViews",
        description="declares its input views",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class DeclaredViews(CellLineFeaturizer):
        input_views = ("methylation",)

        def fit(self, features, *, entity_ids=None, context=None):
            return self

        def transform(self, features, entity_ids):
            raise NotImplementedError

        @property
        def output_dim(self) -> int:
            return 0

    assert get_cell_line_featurizer("declaredViews").resolve_input_views() == ("methylation",)
