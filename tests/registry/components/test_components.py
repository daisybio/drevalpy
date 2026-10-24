"""Tests for the shared component-registry helpers: abstract checks, registration metadata and contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.registry.components import ComponentRegistry
from drevalpy.registry.components._abstract import abstract_members, validate_no_abstract_methods
from drevalpy.registry.components._contract_assignment import assign_contract, declared_contract
from drevalpy.registry.components._metadata import (
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
from drevalpy.types.enums.literature_reference import LiteratureReference


def test_required_fields_are_explicit_per_registry() -> None:
    assert ComponentRegistry._required_fields == ("description",)
    assert FeaturizerRegistry._required_fields == ("description", "contract")
    assert PredictorRegistry._required_fields == (
        "description",
        "sample_contract",
        "drug_contract",
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
    contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    lit = type("Lit", (), {"sample_contract": contract, "drug_contract": contract})
    reference = LiteratureReference(
        repo_url="https://github.com/example/repo", citation_doi="10.1234/example", deviations="none"
    )

    apply_registration_metadata(lit, normalize_registration_metadata("lit model", reference=reference))

    validate_registered_class("predictor", "lit", lit, required_fields=PredictorRegistry._required_fields)


@pytest.mark.parametrize(
    ("role", "required_fields", "attrs", "message"),
    [
        ("drug_featurizer", FeaturizerRegistry._required_fields, {}, r"missing=\['contract'\]"),
        (
            "predictor",
            PredictorRegistry._required_fields,
            {},
            r"missing=\['sample_contract', 'drug_contract'\]",
        ),
        ("predictor", ("description",), {"description": None}, r"missing=\['description'\]"),
        (
            "drug_featurizer",
            FeaturizerRegistry._required_fields,
            {"contract": "numeric_matrix"},
            r"invalid=\['contract'\]",
        ),
    ],
    ids=["featurizer-contract", "predictor-contracts", "description", "wrong-type-contract"],
)
def test_role_validation_reports_missing_and_invalid_fields(role, required_fields, attrs, message) -> None:
    cls = _component(**attrs)
    if attrs.get("description", "x") is None:
        del cls.description  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match=message):
        validate_registered_class(role, "native", cls, required_fields=required_fields)


def test_register_rejects_a_blank_description_and_a_bad_contract_before_the_class_body() -> None:
    registry = FeaturizerRegistry("test", "Test", "tests")

    with pytest.raises(ValueError, match="description must be a non-empty string"):
        registry.register("blank", description="  ", contract=FeatureFormat.NUMERIC_MATRIX)
    with pytest.raises(Exception, match="FeatureContract|FeatureFormat|did not match any element in the union"):
        registry.register("badContract", description="demo", contract="numeric_matrix")  # type: ignore[arg-type]


GRAPH = FeatureContract(format=FeatureFormat.GRAPH)


MATRIX = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def _class_with(base: type = object, **attrs) -> type:
    return type("Component", (base,), attrs)


class TestAssignContract:
    @pytest.mark.parametrize(
        ("body", "argument", "expected"),
        [
            ({}, GRAPH, GRAPH),
            ({"contract": MATRIX}, GRAPH, GRAPH),
            ({"contract": GRAPH}, None, GRAPH),
            ({"contract": FeatureFormat.GRAPH}, None, GRAPH),
        ],
        ids=["decorator-only", "decorator-wins", "class-body-fallback", "format-shorthand"],
    )
    def test_resolution_order(self, body: dict, argument, expected: FeatureContract) -> None:
        component = _class_with(**body)

        assign_contract(component, "contract", argument)

        assert component.contract == expected

    def test_a_declaration_inherited_from_a_base_class_is_usable(self) -> None:
        component = _class_with(_class_with(contract=MATRIX))

        assign_contract(component, "contract", None)

        assert component.__dict__["contract"] == MATRIX

    def test_assigns_each_named_attribute_independently(self) -> None:
        component = _class_with()

        assign_contract(component, "sample_contract", MATRIX)
        assign_contract(component, "drug_contract", GRAPH)

        assert (component.sample_contract, component.drug_contract) == (MATRIX, GRAPH)

    def test_rejects_a_component_that_declares_nothing_naming_the_class_and_both_fixes(self) -> None:
        with pytest.raises(ValueError, match="no sample_contract declared"):
            assign_contract(_class_with(), "sample_contract", None)
        with pytest.raises(ValueError, match=r"Component: .*pass contract= to the .*or set it on the class body"):
            assign_contract(_class_with(), "contract", None)


class TestDeclaredContract:
    def test_returns_a_contract_and_promotes_a_bare_format(self) -> None:
        assert declared_contract(_class_with(contract=GRAPH), "contract") is GRAPH
        assert declared_contract(_class_with(contract=FeatureFormat.NUMERIC_MATRIX), "contract") == MATRIX

    def test_an_explicit_none_counts_as_undeclared(self) -> None:
        with pytest.raises(ValueError, match="no contract declared"):
            declared_contract(_class_with(contract=None), "contract")

    @pytest.mark.parametrize("value", ["numeric_matrix", 42])
    def test_an_unusable_declaration_is_invalid_not_missing_and_keeps_the_type_error_as_cause(self, value) -> None:
        with pytest.raises(ValueError, match="class-body contract is invalid") as excinfo:
            declared_contract(_class_with(contract=value), "contract")

        assert isinstance(excinfo.value.__cause__, TypeError)


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
