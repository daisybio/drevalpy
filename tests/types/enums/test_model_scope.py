"""Tests for the ``ModelScope`` training-scope enum."""

from __future__ import annotations

from enum import StrEnum

import pytest

from drevalpy.types.enums.model_scope import ModelScope


class TestMembers:
    def test_exactly_two_scopes_exist(self):
        assert set(ModelScope) == {ModelScope.MULTI_DRUG, ModelScope.SINGLE_DRUG}

    @pytest.mark.parametrize(
        ("member", "value"),
        [
            pytest.param(ModelScope.MULTI_DRUG, "multi_drug", id="multi-drug"),
            pytest.param(ModelScope.SINGLE_DRUG, "single_drug", id="single-drug"),
        ],
    )
    def test_values_are_the_serialized_names(self, member, value):
        assert member.value == value

    def test_lookup_by_value_returns_the_member(self):
        assert ModelScope("single_drug") is ModelScope.SINGLE_DRUG

    def test_an_unknown_value_is_rejected(self):
        with pytest.raises(ValueError, match="per_tissue"):
            ModelScope("per_tissue")


class TestStringBehaviour:
    def test_members_are_strings(self):
        assert isinstance(ModelScope.MULTI_DRUG, str)
        assert issubclass(ModelScope, StrEnum)

    def test_members_compare_equal_to_their_value(self):
        assert ModelScope.MULTI_DRUG == "multi_drug"

    def test_string_formatting_uses_the_value(self):
        assert f"{ModelScope.SINGLE_DRUG}" == "single_drug"

    def test_members_are_usable_as_mapping_keys_alongside_strings(self):
        registry = {ModelScope.MULTI_DRUG: "global"}

        assert registry["multi_drug"] == "global"
