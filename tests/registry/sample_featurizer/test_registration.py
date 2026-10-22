"""Tests for public sample featurizer registration and lookup helpers."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.registry.sample_featurizer import (
    get as get_sample_featurizer,
)
from drevalpy.registry.sample_featurizer import (
    list as list_sample_featurizers,
)
from drevalpy.registry.sample_featurizer import (
    metadata as get_sample_featurizer_metadata,
)
from drevalpy.registry.sample_featurizer import (
    register as register_sample_featurizer,
)
from tests.registry._helpers import isolated_component_registries


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    yield from isolated_component_registries()


def test_register_and_lookup_sample_featurizer() -> None:
    @register_sample_featurizer(
        "dummySample",
        description="test sample",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class DummySample:
        pass

    assert get_sample_featurizer("dummySample") is DummySample
    assert vars(DummySample)["contract"] == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    assert "dummySample" in list_sample_featurizers()


def test_duplicate_registration_fails() -> None:
    @register_sample_featurizer(
        "dup",
        description="first",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class First:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @register_sample_featurizer(
            "dup",
            description="second",
            contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class Second:
            pass


def test_unknown_component_fails() -> None:
    with pytest.raises(ValueError, match="Unknown Sample featurizer"):
        get_sample_featurizer("missing")


def test_get_metadata_includes_output_format() -> None:
    @register_sample_featurizer(
        "graphFeat",
        description="graph",
        contract=FeatureFormat.GRAPH,
    )
    class GraphFeat:
        pass

    meta = get_sample_featurizer_metadata("graphFeat")
    assert meta["output_format"] == "graph"
    assert meta["description"] == "graph"
    assert meta["tags"] == frozenset()


def test_duplicate_class_and_decorator_contract_prefers_the_decorator() -> None:
    @register_sample_featurizer(
        "conflict",
        description="conflict",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class Conflict:
        contract = FeatureContract(format=FeatureFormat.GRAPH)

    assert vars(Conflict)["contract"] == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def test_a_class_body_contract_is_a_valid_declaration() -> None:
    @register_sample_featurizer("bodyContract", description="declares on the class body")
    class BodyContract:
        contract = FeatureContract(format=FeatureFormat.GRAPH)

    assert get_sample_featurizer_metadata("bodyContract")["output_format"] == "graph"


def test_a_class_body_format_shorthand_is_normalized() -> None:
    @register_sample_featurizer("shorthandContract", description="format shorthand")
    class ShorthandContract:
        contract = FeatureFormat.RAGGED_SEQUENCE

    assert vars(ShorthandContract)["contract"] == FeatureContract(format=FeatureFormat.RAGGED_SEQUENCE)


def test_a_featurizer_declaring_no_contract_anywhere_is_rejected() -> None:
    with pytest.raises(ValueError, match="no contract declared"):

        @register_sample_featurizer("noContract", description="missing contract")
        class NoContract:
            pass
