"""Tests for the catalog metadata dicts built for registered components.

These are pure functions over class attributes, so every case is a bare class -
no registry involved and therefore no global registry state to restore.
"""

from __future__ import annotations

import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.registry.components._metadata import (
    base_component_metadata,
    featurizer_component_metadata,
    predictor_component_metadata,
)
from drevalpy.types.enums.literature_reference import LiteratureReference

_EMPTY_REFERENCE_FIELDS = {
    "repo_url": "",
    "citation": "",
    "citation_doi": "",
    "citation_text": "",
    "deviations": "",
}


class _Bare:
    pass


class _Described:
    description = "a described component"
    tags = frozenset({"baseline"})


class _WithDoi:
    description = "doi reference"
    reference = LiteratureReference(
        repo_url="https://github.com/example/repo",
        citation_doi="10.1234/example",
        deviations="none",
    )


class _WithCitationText:
    description = "text reference"
    reference = LiteratureReference(
        repo_url="https://github.com/example/repo",
        citation_text="Doe et al., 2024",
    )


class _WithBadReference:
    description = "not a reference"
    reference = "https://example.org/paper"


def test_base_metadata_carries_registry_and_name() -> None:
    meta = base_component_metadata("predictors", "demo", _Described)

    assert meta["registry"] == "predictors"
    assert meta["name"] == "demo"
    assert meta["class_name"] == "_Described"


def test_base_metadata_copies_description_and_tags() -> None:
    meta = base_component_metadata("predictors", "demo", _Described)

    assert meta["description"] == "a described component"
    assert meta["tags"] == frozenset({"baseline"})


def test_base_metadata_defaults_missing_description_and_tags() -> None:
    meta = base_component_metadata("predictors", "bare", _Bare)

    assert meta["description"] == ""
    assert meta["tags"] == frozenset()


def test_base_metadata_coerces_a_none_description() -> None:
    class NoneDescription:
        description = None

    meta = base_component_metadata("predictors", "none", NoneDescription)

    assert meta["description"] == ""


def test_base_metadata_leaves_reference_fields_empty_without_a_reference() -> None:
    meta = base_component_metadata("predictors", "bare", _Bare)

    assert {key: meta[key] for key in _EMPTY_REFERENCE_FIELDS} == _EMPTY_REFERENCE_FIELDS


def test_base_metadata_ignores_a_reference_of_the_wrong_type() -> None:
    meta = base_component_metadata("predictors", "bad", _WithBadReference)

    assert {key: meta[key] for key in _EMPTY_REFERENCE_FIELDS} == _EMPTY_REFERENCE_FIELDS


def test_base_metadata_expands_a_doi_into_a_resolvable_url() -> None:
    meta = base_component_metadata("predictors", "doi", _WithDoi)

    assert meta["citation"] == "https://doi.org/10.1234/example"
    assert meta["citation_doi"] == "10.1234/example"


def test_base_metadata_copies_the_reference_repo_and_deviations() -> None:
    meta = base_component_metadata("predictors", "doi", _WithDoi)

    assert meta["repo_url"] == "https://github.com/example/repo"
    assert meta["deviations"] == "none"


def test_base_metadata_falls_back_to_the_citation_text() -> None:
    meta = base_component_metadata("predictors", "text", _WithCitationText)

    assert meta["citation"] == "Doe et al., 2024"
    assert meta["citation_text"] == "Doe et al., 2024"


def test_featurizer_metadata_reports_the_contract_format() -> None:
    class GraphFeaturizer:
        description = "graph featurizer"
        contract = FeatureContract(format=FeatureFormat.GRAPH)

    meta = featurizer_component_metadata("drug_featurizers", "graph", GraphFeaturizer)

    assert meta["output_format"] == "graph"


def test_featurizer_metadata_defaults_precompute_to_false() -> None:
    class NumericFeaturizer:
        description = "numeric featurizer"
        contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

    meta = featurizer_component_metadata("drug_featurizers", "numeric", NumericFeaturizer)

    assert meta["precompute"] is False


def test_featurizer_metadata_reports_an_opted_in_precompute() -> None:
    class PrecomputedFeaturizer:
        description = "precomputed featurizer"
        contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
        precompute = True

    meta = featurizer_component_metadata("drug_featurizers", "precomputed", PrecomputedFeaturizer)

    assert meta["precompute"] is True


def test_featurizer_metadata_keeps_the_shared_base_fields() -> None:
    class NumericFeaturizer:
        description = "numeric featurizer"
        contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

    meta = featurizer_component_metadata("drug_featurizers", "numeric", NumericFeaturizer)

    assert meta["registry"] == "drug_featurizers"
    assert meta["description"] == "numeric featurizer"


def test_featurizer_metadata_requires_a_contract() -> None:
    with pytest.raises(TypeError, match="must define a contract"):
        featurizer_component_metadata("drug_featurizers", "bare", _Bare)


def test_predictor_metadata_reports_the_input_interface() -> None:
    class FeatureFreeLike:
        description = "feature free"
        input_interface = "feature_free"

    meta = predictor_component_metadata("predictors", "featureFree", FeatureFreeLike)

    assert meta["input_interface"] == "feature_free"


def test_predictor_metadata_defaults_the_input_interface_to_empty() -> None:
    meta = predictor_component_metadata("predictors", "bare", _Bare)

    assert meta["input_interface"] == ""


def test_predictor_metadata_keeps_the_shared_base_fields() -> None:
    meta = predictor_component_metadata("predictors", "demo", _Described)

    assert meta["registry"] == "predictors"
    assert meta["tags"] == frozenset({"baseline"})
