"""Tests for the featurizer class-body declarations (``_declarations``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.featurizers._declarations import FeaturizerDeclarationsMixin
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.types.data.batch.feature_block import FeaturizerResult
from tests.components.featurizers._helpers import DoublingFeaturizer


class _Declared(Featurizer):
    entity_id_only = True

    def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
        return self

    def _transform_features(self, source, entity_ids) -> FeaturizerResult:
        return FeaturizerResult.from_blocks({}, len(entity_ids))

    @property
    def output_dim(self) -> int:
        return 0


_Declared.contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def test_the_mixin_is_part_of_the_featurizer_base() -> None:
    assert issubclass(DoublingFeaturizer, FeaturizerDeclarationsMixin)


def test_contract_declarations_are_normalized_at_class_creation() -> None:
    class _Shorthand(Featurizer):
        contract = FeatureFormat.GRAPH

    class _Inheriting(DoublingFeaturizer):
        pass

    assert _Shorthand.contract == FeatureContract(format=FeatureFormat.GRAPH)
    assert _Inheriting.contract is DoublingFeaturizer.contract
    with pytest.raises(TypeError, match="class-body contract is invalid"):

        class _Bad(Featurizer):
            contract = "graph_but_a_plain_string"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({"view": "mutations"}, ("mutations",)), ({"view": "   "}, ("test_view",)), ({}, ("test_view",))],
)
def test_resolve_input_views(kwargs, expected) -> None:
    assert DoublingFeaturizer.resolve_input_views(**kwargs) == expected
    assert _Declared.resolve_input_views() == ()


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        ({"entity_id_only": False, "requires_view": True}, "requires an explicit view"),
        ({"entity_id_only": False}, "declare input_views on the class body"),
    ],
)
def test_resolve_input_views_rejects_undeclared_views(attrs, match) -> None:
    cls = type("_Bad", (_Declared,), attrs)
    with pytest.raises(TypeError, match=match):
        cls.resolve_input_views()


@pytest.mark.parametrize(("config_view", "expected"), [(None, "test_view"), ("mutations", "mutations")])
def test_output_block_specs_follow_the_config_view_and_contract(config_view, expected) -> None:
    specs = DoublingFeaturizer.output_block_specs_for_config(SimpleNamespace(view=config_view))
    assert [spec.name for spec in specs] == [expected]
    assert [spec.format for spec in specs] == [FeatureFormat.NUMERIC_MATRIX]
    assert _Declared.output_block_specs_for_config(None) == ()
