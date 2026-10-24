"""Type enums and small value types: ModelScope, PredictionMode and LiteratureReference."""

from __future__ import annotations

from enum import StrEnum

import pytest

import drevalpy.models.config as model_config
import drevalpy.types.enums.prediction_mode as prediction_mode_module
from drevalpy.types.enums.literature_reference import LiteratureReference
from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.types.enums.prediction_mode import PredictionMode


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


def test_prediction_mode_values() -> None:
    assert PredictionMode.REGRESSION == "regression"
    assert PredictionMode.CLASSIFICATION == "classification"


def test_prediction_mode_reexported_from_models_config() -> None:
    assert model_config.PredictionMode is PredictionMode


def test_predictors_base_imports_prediction_mode_from_types() -> None:
    import drevalpy.components.predictors.abstract.base as base_module

    source_path = base_module.__file__
    assert source_path is not None
    text = open(source_path, encoding="utf-8").read()
    assert "drevalpy.types.enums.prediction_mode" in text
    assert "drevalpy.models.config" not in text


def test_prediction_mode_module_has_no_models_dependency() -> None:
    source_path = prediction_mode_module.__file__
    assert source_path is not None
    text = open(source_path, encoding="utf-8").read()
    assert "drevalpy.models" not in text


def test_literature_reference_strips_fields() -> None:
    ref = LiteratureReference(
        repo_url="  https://github.com/example/repo  ",
        citation_doi=" 10.1234/example ",
        citation_text=" Example paper. ",
        deviations=" none ",
    )
    assert ref.repo_url == "https://github.com/example/repo"
    assert ref.citation_doi == "10.1234/example"
    assert ref.citation_text == "Example paper."
    assert ref.deviations == "none"
