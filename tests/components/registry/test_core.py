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
    list_cell_line_featurizer_metadata,
    list_cell_line_featurizers,
    register_cell_line_featurizer,
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
    @register_cell_line_featurizer("dummyCellLine", description="test cell line", category="native")
    class DummyCellLine:
        output_contract = FeatureContract(kind=FeatureKind.DENSE)

    assert get_cell_line_featurizer("dummyCellLine") is DummyCellLine
    assert "dummyCellLine" in list_cell_line_featurizers()


def test_duplicate_registration_fails() -> None:
    @register_cell_line_featurizer("dup", description="first", category="native")
    class First:
        output_contract = FeatureContract(kind=FeatureKind.DENSE)

    with pytest.raises(ValueError, match="already registered"):

        @register_cell_line_featurizer("dup", description="second", category="native")
        class Second:
            output_contract = FeatureContract(kind=FeatureKind.DENSE)


def test_unknown_component_fails() -> None:
    with pytest.raises(ValueError, match="Unknown Cell line featurizer"):
        get_cell_line_featurizer("missing")


def test_metadata_listing_and_category_filter() -> None:
    @register_cell_line_featurizer("nativeEnc", description="native", category="native")
    class NativeEnc:
        output_contract = FeatureContract(kind=FeatureKind.DENSE)

    @register_cell_line_featurizer("baselineEnc", description="baseline", category="baseline")
    class BaselineEnc:
        output_contract = FeatureContract(kind=FeatureKind.DENSE)

    all_rows = list_cell_line_featurizer_metadata()
    assert len(all_rows) == 2
    baseline_rows = list_cell_line_featurizer_metadata(category="baseline")
    assert len(baseline_rows) == 1
    assert baseline_rows[0]["name"] == "baselineEnc"


def test_get_metadata_includes_output_type() -> None:
    @register_cell_line_featurizer("graphEnc", description="graph", category="native")
    class GraphEnc:
        output_contract = FeatureContract(kind=FeatureKind.GRAPH, backend="pyg")

    meta = get_cell_line_featurizer_metadata("graphEnc")
    assert meta["output_type"] == "graph"
    assert meta["description"] == "graph"


def test_decorator_returns_original_class() -> None:
    @register_predictor("dummyPred", description="pred", category="baseline")
    class DummyPred:
        uses_features = False
        supported_modes = {"regression"}
        required_cell_line_contract = FeatureContract(kind=FeatureKind.DENSE)
        required_drug_contract = FeatureContract(kind=FeatureKind.DENSE)

    assert vars(DummyPred)["registry_name"] == "dummyPred"


def test_registry_clear() -> None:
    registry = Registry("test", "Test component", "test_components", lambda *_: {})
    decorated = registry.register("x", description="x", category="native")

    @decorated
    class X:
        pass

    assert registry.list_names() == ["x"]
    registry.clear()
    assert registry.list_names() == []
