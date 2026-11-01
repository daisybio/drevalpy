"""Tests for the shared predictor/featurizer registration and their catalog facts."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import pytest

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.registry._component import (
    LiteratureReference,
    component_decorator,
    featurizer_component_metadata,
    predictor_component_metadata,
)
from drevalpy.registry._registry import Registry
from drevalpy.registry.featurizer import featurizer_registry
from drevalpy.registry.featurizer import register as register_featurizer
from drevalpy.registry.predictor import predictor_registry
from drevalpy.types.data.batch.feature_block import BlockSpec
from tests.registry._helpers import empty_component_registries

_REFERENCE_KWARGS = {
    "repo_url": "https://github.com/example/repo",
    "citation_doi": "10.1234/example",
    "deviations": "none",
}


class _Base(ABC):
    @abstractmethod
    def alpha(self) -> None: ...

    @abstractmethod
    def beta(self) -> None: ...


class _Partial(_Base):
    def alpha(self) -> None: ...


class _Plain:
    pass


def _no_checks(name: str, cls: type) -> None:
    pass


def _decorator(registry: Registry[type], name: str = "demo", **kwargs):
    options = {"description": "demo", "tags": None, "reference": None, "validate": _no_checks} | kwargs
    return component_decorator(registry, name, **options)


def test_literature_reference_strips_fields_and_expands_a_doi() -> None:
    ref = LiteratureReference(
        repo_url="  https://github.com/example/repo  ",
        citation_doi=" 10.1234/example ",
        citation_text=" Example paper. ",
        deviations=" none ",
    )

    assert (ref.repo_url, ref.citation_doi, ref.citation_text, ref.deviations) == (
        "https://github.com/example/repo",
        "10.1234/example",
        "Example paper.",
        "none",
    )
    assert ref.citation == "https://doi.org/10.1234/example"


def test_literature_reference_falls_back_to_the_citation_text() -> None:
    ref = LiteratureReference(repo_url="https://github.com/example/repo", citation_text="Doe et al.", deviations="x")

    assert ref.citation == "Doe et al."


@pytest.mark.parametrize(
    ("kwargs", "invalid"),
    [
        ({"repo_url": "https://github.com/example/repo"}, "['citation', 'deviations']"),
        ({**_REFERENCE_KWARGS, "repo_url": "github.com/example"}, "['repo_url']"),
        ({**_REFERENCE_KWARGS, "deviations": "  "}, "['deviations']"),
    ],
    ids=["missing-citation-and-deviations", "not-a-url", "blank-deviations"],
)
def test_literature_reference_rejects_missing_or_malformed_fields(kwargs: dict, invalid: str) -> None:
    with pytest.raises(ValueError, match=re.escape(f"invalid fields: {invalid}")):
        LiteratureReference(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"description": "  "}, ValueError, "description must be a non-empty string"),
        ({"tags": "baseline"}, TypeError, "iterable of strings"),
        ({"tags": ("ok", 1)}, TypeError, "tags must contain strings"),
        ({"reference": "not-a-reference"}, TypeError, "reference must be LiteratureReference"),
    ],
    ids=["blank-description", "bare-string-tags", "non-string-tags", "bad-reference-type"],
)
def test_bad_registration_arguments_are_rejected_before_the_class_body(
    kwargs: dict, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _decorator(Registry("example"), **kwargs)


def test_registration_stores_description_and_tags_in_the_registry_and_the_reference_on_the_class() -> None:
    registry: Registry[type] = Registry("example")
    reference = LiteratureReference(**_REFERENCE_KWARGS)
    lit = type("Lit", (), {})

    returned = _decorator(registry, "lit", description="  lit model ", tags=(" b ", ""), reference=reference)(lit)

    assert returned is lit
    assert registry.get("lit") is lit
    assert dict(registry.metadata("lit")) == {"description": "lit model", "tags": frozenset({"b"})}
    assert vars(lit)["reference"] is reference
    assert not {"description", "tags", "registry_name"} & set(vars(lit))


def test_an_incomplete_subclass_is_rejected_before_the_kind_validator() -> None:
    registry: Registry[type] = Registry("predictor")
    calls: list[str] = []

    with pytest.raises(ValueError, match=r"predictor 'partial' \(_Partial\) does not implement beta"):
        _decorator(registry, "partial", validate=lambda n, c: calls.append(n))(_Partial)
    with pytest.raises(ValueError, match="does not implement alpha, beta.*register a concrete subclass"):
        _decorator(registry, "base")(_Base)

    assert calls == []
    assert len(registry) == 0


def test_nothing_is_registered_when_the_kind_validator_fails() -> None:
    registry: Registry[type] = Registry("example")

    def reject(name: str, cls: type) -> None:
        raise ValueError(f"bad {name}")

    with pytest.raises(ValueError, match="bad rejected"):
        _decorator(registry, "rejected", validate=reject)(_Plain)

    assert "rejected" not in registry
    assert not hasattr(_Plain, "reference")


def test_a_duplicate_name_is_rejected() -> None:
    registry: Registry[type] = Registry("example")
    _decorator(registry, "dup")(type("First", (), {}))

    with pytest.raises(ValueError, match="already registered"):
        _decorator(registry, "dup")(type("Second", (), {}))


def test_register_rejects_a_blank_description_before_the_class_body() -> None:
    with pytest.raises(ValueError, match="description must be a non-empty string"):
        register_featurizer("blank", side="sample", description="  ")


def _featurizer(output_format: FeatureFormat | None = FeatureFormat.NUMERIC_MATRIX, **attrs):
    def output_blocks(cls, config):
        if output_format is None:
            raise ValueError("config-dependent")
        return (BlockSpec("a", output_format), BlockSpec("b", output_format))

    return type("Featurizer", (), {"side": "drug", "output_blocks": classmethod(output_blocks)} | attrs)


@pytest.fixture
def _empty_registries() -> None:
    empty_component_registries()


@pytest.mark.usefixtures("_empty_registries")
@pytest.mark.parametrize(
    ("kwargs", "output_format"),
    [
        ({"output_format": FeatureFormat.GRAPH}, "graph"),
        ({}, "numeric_matrix"),
        ({"output_format": None}, ""),
    ],
    ids=["graph", "default", "config-dependent-blocks"],
)
def test_featurizer_metadata(kwargs: dict, output_format: str) -> None:
    cls = _featurizer(**kwargs)
    featurizer_registry("drug").register("f", cls, metadata={"description": "d", "tags": frozenset({"t"})})

    meta = featurizer_component_metadata("featurizer", "f", cls)

    assert meta["output_format"] == output_format
    assert "precompute" not in meta
    assert (meta["registry"], meta["name"], meta["class_name"]) == ("featurizer", "f", "Featurizer")
    assert (meta["description"], meta["tags"], meta["side"]) == ("d", frozenset({"t"}), "drug")
    assert (meta["repo_url"], meta["citation"], meta["deviations"]) == ("", "", "")


@pytest.mark.usefixtures("_empty_registries")
@pytest.mark.parametrize(
    ("sample", "drug", "interface"),
    [
        (Inputs.none(), Inputs.none(), "feature_free"),
        (Inputs.matrix(), Inputs.none(), "matrix"),
        (Inputs.matrix(), Inputs.blocks(), "block"),
    ],
    ids=["feature-free", "matrix", "block"],
)
def test_predictor_metadata_summarizes_the_inputs_and_copies_the_reference(
    sample: Inputs, drug: Inputs, interface: str
) -> None:
    reference = LiteratureReference(**_REFERENCE_KWARGS)
    cls = type("Declared", (), {"sample_inputs": sample, "drug_inputs": drug, "reference": reference})
    predictor_registry.register("declared", cls, metadata={"description": "p", "tags": frozenset()})

    meta = predictor_component_metadata("predictors", "declared", cls)

    assert meta["input_interface"] == interface
    assert (meta["registry"], meta["description"], meta["tags"]) == ("predictors", "p", frozenset())
    assert meta["citation"] == "https://doi.org/10.1234/example"
    assert (meta["citation_doi"], meta["repo_url"], meta["deviations"]) == (
        "10.1234/example",
        "https://github.com/example/repo",
        "none",
    )
