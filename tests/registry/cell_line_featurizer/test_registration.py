"""Tests for public cell-line featurizer registration and lookup helpers."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.registry.cell_line_featurizer import (
    get as get_cell_line_featurizer,
)
from drevalpy.registry.cell_line_featurizer import (
    list as list_cell_line_featurizers,
)
from drevalpy.registry.cell_line_featurizer import (
    metadata as get_cell_line_featurizer_metadata,
)
from drevalpy.registry.cell_line_featurizer import (
    register as register_cell_line_featurizer,
)
from tests.registry._helpers import isolated_component_registries


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    yield from isolated_component_registries()


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


def test_unknown_component_fails() -> None:
    with pytest.raises(ValueError, match="Unknown Cell line featurizer"):
        get_cell_line_featurizer("missing")


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


def test_duplicate_class_and_decorator_contract_fails() -> None:
    with pytest.raises(ValueError, match="do not set contract on the class body"):

        @register_cell_line_featurizer(
            "conflict",
            description="conflict",
            contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class Conflict:
            contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
