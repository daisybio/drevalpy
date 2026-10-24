"""Tests for the catalog metadata dicts built for registered components (pure functions over bare classes)."""

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


def _featurizer(**attrs):
    return type("Featurizer", (), {"contract": FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)} | attrs)


def test_base_metadata_carries_registry_name_description_and_tags() -> None:
    meta = base_component_metadata("predictors", "demo", _Described)

    assert (meta["registry"], meta["name"], meta["class_name"]) == ("predictors", "demo", "_Described")
    assert (meta["description"], meta["tags"]) == ("a described component", frozenset({"baseline"}))


@pytest.mark.parametrize("cls", [_Bare, type("NoneDescription", (), {"description": None})], ids=["missing", "none"])
def test_base_metadata_defaults_missing_description_and_tags(cls: type) -> None:
    meta = base_component_metadata("predictors", "bare", cls)

    assert (meta["description"], meta["tags"]) == ("", frozenset())
    assert {key: meta[key] for key in _EMPTY_REFERENCE_FIELDS} == _EMPTY_REFERENCE_FIELDS


def test_base_metadata_ignores_a_reference_of_the_wrong_type() -> None:
    meta = base_component_metadata("predictors", "bad", type("Bad", (), {"reference": "https://example.org/paper"}))

    assert {key: meta[key] for key in _EMPTY_REFERENCE_FIELDS} == _EMPTY_REFERENCE_FIELDS


def test_base_metadata_expands_a_doi_into_a_resolvable_url_and_copies_repo_and_deviations() -> None:
    reference = LiteratureReference(
        repo_url="https://github.com/example/repo", citation_doi="10.1234/example", deviations="none"
    )

    meta = base_component_metadata("predictors", "doi", type("WithDoi", (), {"reference": reference}))

    assert meta["citation"] == "https://doi.org/10.1234/example"
    assert meta["citation_doi"] == "10.1234/example"
    assert (meta["repo_url"], meta["deviations"]) == ("https://github.com/example/repo", "none")


def test_base_metadata_falls_back_to_the_citation_text() -> None:
    reference = LiteratureReference(repo_url="https://github.com/example/repo", citation_text="Doe et al., 2024")

    meta = base_component_metadata("predictors", "text", type("WithText", (), {"reference": reference}))

    assert meta["citation"] == meta["citation_text"] == "Doe et al., 2024"


@pytest.mark.parametrize(
    ("attrs", "output_format", "precompute"),
    [
        ({"contract": FeatureContract(format=FeatureFormat.GRAPH)}, "graph", False),
        ({}, "numeric_matrix", False),
        ({"precompute": True}, "numeric_matrix", True),
    ],
    ids=["graph", "default-precompute", "opted-in-precompute"],
)
def test_featurizer_metadata(attrs: dict, output_format: str, precompute: bool) -> None:
    meta = featurizer_component_metadata("drug_featurizers", "f", _featurizer(description="d", **attrs))

    assert (meta["output_format"], meta["precompute"]) == (output_format, precompute)
    assert (meta["registry"], meta["description"]) == ("drug_featurizers", "d")


def test_featurizer_metadata_requires_a_contract() -> None:
    with pytest.raises(TypeError, match="must define a contract"):
        featurizer_component_metadata("drug_featurizers", "bare", _Bare)


def test_predictor_metadata_reports_the_input_interface_and_keeps_the_base_fields() -> None:
    free = type("FeatureFreeLike", (_Described,), {"input_interface": "feature_free"})

    meta = predictor_component_metadata("predictors", "featureFree", free)

    assert meta["input_interface"] == "feature_free"
    assert (meta["registry"], meta["tags"]) == ("predictors", frozenset({"baseline"}))
    assert predictor_component_metadata("predictors", "bare", _Bare)["input_interface"] == ""
