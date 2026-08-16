"""Tests for the featurizer class-body declarations.

Mirrors :mod:`drevalpy.components.featurizers._declarations`, which holds what a
featurizer declares rather than what it computes: the ``contract`` normalization
that runs at class creation, the ``resolve_input_views`` hook the model config
calls to know what to load from disk, and the ``output_block_specs_for_config``
hook ``models/config/_block_specs.py`` reads to predict an output shape.

None of it needs an instance, so every case below asserts against a class.
"""

from __future__ import annotations

import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.featurizers._declarations import FeaturizerDeclarationsMixin
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.types.data.batch.feature_block import FeatureBlock
from tests.components.featurizers._helpers import DoublingFeaturizer


class _Declared(Featurizer):
    """Minimal concrete featurizer, for the class-level hooks."""

    entity_id_only = True

    def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
        return self

    def _transform_blocks(self, source, entity_ids) -> dict[str, FeatureBlock]:
        return {}

    @property
    def output_dim(self) -> int:
        return 0


_Declared.contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def test_the_mixin_is_part_of_the_featurizer_base() -> None:
    """The declarations are reached through ``Featurizer``, not wired in per subclass."""
    assert issubclass(DoublingFeaturizer, FeaturizerDeclarationsMixin)


class TestContractNormalization:
    """``__init_subclass__`` widens a class-body declaration at class creation."""

    def test_a_class_body_contract_is_kept(self) -> None:
        class _BodyContract(Featurizer):  # noqa: B903
            contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

        assert _BodyContract.contract == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

    def test_a_format_shorthand_is_widened(self) -> None:
        class _Shorthand(Featurizer):  # noqa: B903
            contract = FeatureFormat.GRAPH

        assert _Shorthand.contract == FeatureContract(format=FeatureFormat.GRAPH)

    def test_an_invalid_declaration_is_rejected_at_class_creation(self) -> None:
        with pytest.raises(TypeError, match="class-body contract is invalid"):

            class _Bad(Featurizer):  # noqa: B903
                contract = "graph_but_a_plain_string"

    def test_a_subclass_declaring_nothing_inherits_the_contract(self) -> None:
        """Registration supplies the contract for these; nothing is overwritten."""

        class _Inheriting(DoublingFeaturizer):
            pass

        assert _Inheriting.contract is DoublingFeaturizer.contract


class TestResolveInputViews:
    """Which raw views the model config has to load from disk for this featurizer."""

    def test_an_explicit_view_wins(self) -> None:
        assert DoublingFeaturizer.resolve_input_views(view="mutations") == ("mutations",)

    def test_a_blank_view_falls_back_to_the_declaration(self) -> None:
        assert DoublingFeaturizer.resolve_input_views(view="   ") == ("test_view",)

    def test_declared_input_views_are_used_without_kwargs(self) -> None:
        assert DoublingFeaturizer.resolve_input_views() == ("test_view",)

    def test_an_entity_id_only_featurizer_needs_no_views(self) -> None:
        assert _Declared.resolve_input_views() == ()

    def test_a_view_parameterized_featurizer_insists_on_a_view(self) -> None:
        class _NeedsView(_Declared):
            entity_id_only = False
            requires_view = True

        with pytest.raises(TypeError, match="requires an explicit view"):
            _NeedsView.resolve_input_views()

    def test_a_featurizer_declaring_nothing_is_rejected(self) -> None:
        class _Undeclared(_Declared):
            entity_id_only = False

        with pytest.raises(TypeError, match="declare input_views on the class body"):
            _Undeclared.resolve_input_views()


class TestOutputBlockSpecsForConfig:
    """The block names and formats a featurizer will emit under a config node."""

    def test_it_falls_back_to_the_declared_input_view(self) -> None:
        class _Config:
            view = None

        specs = DoublingFeaturizer.output_block_specs_for_config(_Config())

        assert [spec.name for spec in specs] == ["test_view"]

    def test_an_explicit_config_view_wins(self) -> None:
        class _Config:
            view = "mutations"

        specs = DoublingFeaturizer.output_block_specs_for_config(_Config())

        assert [spec.name for spec in specs] == ["mutations"]

    def test_it_carries_the_contract_format(self) -> None:
        specs = DoublingFeaturizer.output_block_specs_for_config(None)

        assert [spec.format for spec in specs] == [FeatureFormat.NUMERIC_MATRIX]

    def test_it_is_empty_without_any_view(self) -> None:
        assert _Declared.output_block_specs_for_config(None) == ()
