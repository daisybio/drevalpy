"""Tests for the shared component-registry helpers: abstract checks and registration metadata."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pytest

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.registry._registry import Registry
from drevalpy.registry.components._abstract import abstract_members, validate_no_abstract_methods
from drevalpy.registry.components._metadata import (
    LiteratureReference,
    base_component_metadata,
    featurizer_component_metadata,
    predictor_component_metadata,
)
from drevalpy.registry.components._register import register_component
from drevalpy.registry.components._registration_metadata import (
    RegistrationMetadata,
    apply_registration_metadata,
    normalize_registration_metadata,
)
from drevalpy.registry.featurizer import register as register_featurizer
from drevalpy.types.data.batch.feature_block import BlockSpec


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
        validate_no_abstract_methods("featurizer", "base", _Base)


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


def _no_checks(name: str, cls: type) -> None:
    pass


def test_register_component_stamps_metadata_registers_and_names_the_class() -> None:
    registry: Registry[type] = Registry("example")
    reference = LiteratureReference(
        repo_url="https://github.com/example/repo", citation_doi="10.1234/example", deviations="none"
    )
    lit = type("Lit", (), {})

    returned = register_component(
        registry, "predictor", "lit", lit, normalize_registration_metadata("lit model", ("b",), reference), _no_checks
    )

    assert returned is lit
    assert registry.get("lit") is lit
    assert dict(registry.metadata("lit")) == {"description": "lit model", "tags": frozenset({"b"})}
    assert (lit.description, lit.reference, lit.registry_name) == ("lit model", reference, "lit")  # type: ignore[attr-defined]


def test_register_component_checks_abstract_members_before_the_kind_validator() -> None:
    registry: Registry[type] = Registry("example")
    calls: list[str] = []

    with pytest.raises(ValueError, match="does not implement beta"):
        register_component(
            registry,
            "predictor",
            "partial",
            _Partial,
            normalize_registration_metadata("p"),
            lambda n, c: calls.append(n),
        )

    assert calls == []
    assert "partial" not in registry


def test_register_component_registers_nothing_when_the_kind_validator_fails() -> None:
    registry: Registry[type] = Registry("example")

    def reject(name: str, cls: type) -> None:
        raise ValueError(f"bad {name}")

    with pytest.raises(ValueError, match="bad rejected"):
        register_component(registry, "predictor", "rejected", _Plain, normalize_registration_metadata("r"), reject)

    assert "rejected" not in registry
    assert not hasattr(_Plain, "registry_name")


def test_register_component_rejects_a_duplicate_name() -> None:
    registry: Registry[type] = Registry("example")
    metadata = normalize_registration_metadata("d")
    register_component(registry, "predictor", "dup", type("First", (), {}), metadata, _no_checks)

    with pytest.raises(ValueError, match="already registered"):
        register_component(registry, "predictor", "dup", type("Second", (), {}), metadata, _no_checks)


def test_register_rejects_a_blank_description_before_the_class_body() -> None:
    with pytest.raises(ValueError, match="description must be a non-empty string"):
        register_featurizer("blank", description="  ")


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


def _featurizer(output_format: FeatureFormat | None = FeatureFormat.NUMERIC_MATRIX, **attrs):
    def output_blocks(cls, config):
        if output_format is None:
            raise ValueError("config-dependent")
        return (BlockSpec("a", output_format), BlockSpec("b", output_format))

    return type("Featurizer", (), {"sides": frozenset({"drug"}), "output_blocks": classmethod(output_blocks)} | attrs)


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
    ("kwargs", "output_format", "precompute"),
    [
        ({"output_format": FeatureFormat.GRAPH}, "graph", False),
        ({}, "numeric_matrix", False),
        ({"precompute": True}, "numeric_matrix", True),
        ({"output_format": None}, "", False),
    ],
    ids=["graph", "default-precompute", "opted-in-precompute", "config-dependent-blocks"],
)
def test_featurizer_metadata(kwargs: dict, output_format: str, precompute: bool) -> None:
    meta = featurizer_component_metadata("featurizer", "f", _featurizer(description="d", **kwargs))

    assert (meta["output_format"], meta["precompute"]) == (output_format, precompute)
    assert (meta["registry"], meta["description"], meta["sides"]) == ("featurizer", "d", frozenset({"drug"}))


@pytest.mark.parametrize(
    ("sample", "drug", "interface"),
    [
        (Inputs.none(), Inputs.none(), "feature_free"),
        (Inputs.matrix(), Inputs.none(), "matrix"),
        (Inputs.matrix(), Inputs.blocks(), "block"),
    ],
    ids=["feature-free", "matrix", "block"],
)
def test_predictor_metadata_summarizes_the_declared_inputs_and_keeps_the_base_fields(
    sample: Inputs, drug: Inputs, interface: str
) -> None:
    cls = type("Declared", (_Described,), {"sample_inputs": sample, "drug_inputs": drug})

    meta = predictor_component_metadata("predictors", "declared", cls)

    assert meta["input_interface"] == interface
    assert (meta["registry"], meta["tags"]) == ("predictors", frozenset({"baseline"}))


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
