"""Tests for the shared component-registry helpers: abstract checks, registration metadata and feature formats."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pytest

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.registry.components import ComponentRegistry
from drevalpy.registry.components._abstract import abstract_members, validate_no_abstract_methods
from drevalpy.registry.components._format_assignment import assign_format
from drevalpy.registry.components._metadata import (
    LiteratureReference,
    base_component_metadata,
    featurizer_component_metadata,
    predictor_component_metadata,
)
from drevalpy.registry.components._metadata_validate import validate_registered_class
from drevalpy.registry.components._registration_metadata import (
    RegistrationMetadata,
    apply_registration_metadata,
    normalize_registration_metadata,
)
from drevalpy.registry.featurizer import FeaturizerRegistry
from drevalpy.registry.predictor import PredictorRegistry


def test_required_fields_are_explicit_per_registry() -> None:
    assert ComponentRegistry._required_fields == ("description",)
    assert FeaturizerRegistry._required_fields == ("description", "output_format")
    assert PredictorRegistry._required_fields == (
        "description",
        "sample_input_format",
        "drug_input_format",
    )


class _Base(ABC):
    @abstractmethod
    def alpha(self) -> None: ...

    @abstractmethod
    def beta(self) -> None: ...


class _Partial(_Base):
    def alpha(self) -> None: ...


class _Complete(_Partial):
    def beta(self) -> None: ...


class _Plain:
    pass


def test_abstract_members_are_reported_sorted_and_only_when_unimplemented() -> None:
    assert abstract_members(_Plain) == ()
    assert abstract_members(_Complete) == ()
    assert abstract_members(_Base) == ("alpha", "beta")
    assert abstract_members(_Partial) == ("beta",)


def test_complete_classes_pass_validation() -> None:
    validate_no_abstract_methods("predictor", "plain", _Plain)
    validate_no_abstract_methods("predictor", "complete", _Complete)


def test_an_incomplete_subclass_is_rejected_naming_every_missing_member_and_the_fix() -> None:
    with pytest.raises(ValueError, match=r"predictor 'partial' \(_Partial\) does not implement beta"):
        validate_no_abstract_methods("predictor", "partial", _Partial)
    with pytest.raises(ValueError, match="does not implement alpha, beta.*register a concrete subclass"):
        validate_no_abstract_methods("sample_featurizer", "base", _Base)


def test_normalize_strips_tags_and_description() -> None:
    reference = LiteratureReference(
        repo_url="https://github.com/example/repo", citation_doi="10.1234/example", deviations="none"
    )

    metadata = normalize_registration_metadata("  demo  ", tags=("  baseline ", "", "omics"), reference=reference)

    assert metadata == RegistrationMetadata(
        description="demo", tags=frozenset({"baseline", "omics"}), reference=reference
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"description": "  "}, ValueError, "description must be a non-empty string"),
        ({"description": "demo", "tags": "baseline"}, TypeError, "iterable of strings"),
        ({"description": "demo", "tags": ("ok", 1)}, TypeError, "tags must contain strings"),
        (
            {"description": "demo", "reference": "not-a-reference"},
            Exception,
            "LiteratureReference|did not match any element in the union",
        ),
        (
            {"description": "demo", "reference": LiteratureReference(repo_url="https://github.com/example/repo")},
            ValueError,
            "invalid fields",
        ),
    ],
    ids=["blank-description", "bare-string-tags", "non-string-tags", "bad-reference-type", "incomplete-reference"],
)
def test_normalize_rejects_bad_input(kwargs: dict, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        normalize_registration_metadata(**kwargs)


def test_apply_registration_metadata_assigns_fields() -> None:
    target = type("Target", (), {})

    apply_registration_metadata(
        target, RegistrationMetadata(description="demo", tags=frozenset({"baseline"}), reference=None)
    )

    assert (target.description, target.tags, target.reference) == ("demo", frozenset({"baseline"}), None)  # type: ignore[attr-defined]


def _component(**attrs) -> type:
    return type("Native", (), {"description": "native", "tags": frozenset(), "reference": None} | attrs)


def test_literature_reference_is_accepted() -> None:
    lit = type(
        "Lit",
        (),
        {"sample_input_format": FeatureFormat.NUMERIC_MATRIX, "drug_input_format": FeatureFormat.NUMERIC_MATRIX},
    )
    reference = LiteratureReference(
        repo_url="https://github.com/example/repo", citation_doi="10.1234/example", deviations="none"
    )

    apply_registration_metadata(lit, normalize_registration_metadata("lit model", reference=reference))

    validate_registered_class("predictor", "lit", lit, required_fields=PredictorRegistry._required_fields)


@pytest.mark.parametrize(
    ("role", "required_fields", "attrs", "message"),
    [
        ("drug_featurizer", FeaturizerRegistry._required_fields, {}, r"missing=\['output_format'\]"),
        (
            "predictor",
            PredictorRegistry._required_fields,
            {},
            r"missing=\['sample_input_format', 'drug_input_format'\]",
        ),
        ("predictor", ("description",), {"description": None}, r"missing=\['description'\]"),
        (
            "drug_featurizer",
            FeaturizerRegistry._required_fields,
            {"output_format": "numeric_matrix"},
            r"invalid=\['output_format'\]",
        ),
    ],
    ids=["featurizer-format", "predictor-formats", "description", "wrong-type-format"],
)
def test_role_validation_reports_missing_and_invalid_fields(role, required_fields, attrs, message) -> None:
    cls = _component(**attrs)
    if attrs.get("description", "x") is None:
        del cls.description  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match=message):
        validate_registered_class(role, "native", cls, required_fields=required_fields)


def test_register_rejects_a_blank_description_before_the_class_body() -> None:
    registry = FeaturizerRegistry("test", "Test", "tests")

    with pytest.raises(ValueError, match="description must be a non-empty string"):
        registry.register("blank", description="  ", output_format=FeatureFormat.NUMERIC_MATRIX)


GRAPH = FeatureFormat.GRAPH
MATRIX = FeatureFormat.NUMERIC_MATRIX


def _class_with(base: type = object, **attrs) -> type:
    return type("Component", (base,), attrs)


class TestAssignFormat:
    @pytest.mark.parametrize(
        ("body", "argument", "expected"),
        [
            ({}, GRAPH, GRAPH),
            ({"output_format": MATRIX}, GRAPH, GRAPH),
            ({"output_format": GRAPH}, None, GRAPH),
        ],
        ids=["decorator-only", "decorator-wins", "class-body-fallback"],
    )
    def test_resolution_order(self, body: dict, argument, expected: FeatureFormat) -> None:
        component = _class_with(**body)

        assign_format(component, "output_format", argument)

        assert component.output_format == expected

    def test_a_declaration_inherited_from_a_base_class_is_usable(self) -> None:
        component = _class_with(_class_with(output_format=MATRIX))

        assign_format(component, "output_format", None)

        assert component.output_format == MATRIX

    def test_assigns_each_named_attribute_independently(self) -> None:
        component = _class_with()

        assign_format(component, "sample_input_format", MATRIX)
        assign_format(component, "drug_input_format", GRAPH)

        assert (component.sample_input_format, component.drug_input_format) == (MATRIX, GRAPH)

    @pytest.mark.parametrize("body", [{}, {"output_format": None}, {"output_format": "numeric_matrix"}])
    def test_rejects_an_undeclared_or_unusable_format_naming_the_class_and_both_fixes(self, body: dict) -> None:
        with pytest.raises(TypeError, match=r"Component\.output_format must be a FeatureFormat.*output_format="):
            assign_format(_class_with(**body), "output_format", None)

    def test_rejects_a_decorator_argument_that_is_not_a_format(self) -> None:
        with pytest.raises(TypeError, match="must be a FeatureFormat"):
            assign_format(_class_with(), "output_format", "graph")  # type: ignore[arg-type]


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
    return type("Featurizer", (), {"output_format": FeatureFormat.NUMERIC_MATRIX} | attrs)


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
        ({"output_format": FeatureFormat.GRAPH}, "graph", False),
        ({}, "numeric_matrix", False),
        ({"precompute": True}, "numeric_matrix", True),
    ],
    ids=["graph", "default-precompute", "opted-in-precompute"],
)
def test_featurizer_metadata(attrs: dict, output_format: str, precompute: bool) -> None:
    meta = featurizer_component_metadata("drug_featurizers", "f", _featurizer(description="d", **attrs))

    assert (meta["output_format"], meta["precompute"]) == (output_format, precompute)
    assert (meta["registry"], meta["description"]) == ("drug_featurizers", "d")


def test_predictor_metadata_reports_the_input_interface_and_keeps_the_base_fields() -> None:
    free = type("FeatureFreeLike", (_Described,), {"input_interface": "feature_free"})

    meta = predictor_component_metadata("predictors", "featureFree", free)

    assert meta["input_interface"] == "feature_free"
    assert (meta["registry"], meta["tags"]) == ("predictors", frozenset({"baseline"}))
    assert predictor_component_metadata("predictors", "bare", _Bare)["input_interface"] == ""


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
