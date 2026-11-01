"""Parsing of model configs: the recipe grammar and the accepted and rejected featurizer/predictor spellings."""

from __future__ import annotations

import ast
import re
from typing import Any

import pytest

from drevalpy.models.config import FeaturizerConfig
from drevalpy.models.config._parse import (
    expand_featurizer_recipe,
    format_model_recipe,
    normalize_featurizer,
    normalize_predictor,
    parse_featurizer_atoms,
    parse_model_recipe,
)
from drevalpy.registry.featurizer import get_featurizer
from drevalpy.registry.predictor import get as get_predictor


@pytest.mark.parametrize(
    ("recipe", "expected"),
    [
        ("scaledGeneExpression", [("scaledGeneExpression", None)]),
        ("view[copy_number_variation_gistic]", [("view", "copy_number_variation_gistic")]),
        ("fingerprints+oneHot", [("fingerprints", None), ("oneHot", None)]),
        (
            "view[expression]+pca[proteomics]+scaledGeneExpression",
            [("view", "expression"), ("pca", "proteomics"), ("scaledGeneExpression", None)],
        ),
        ("  view[expression] + view[mutations]  ", [("view", "expression"), ("view", "mutations")]),
        ("view[a+b]", [("view", "a+b")]),
        ("notARegisteredName", [("notARegisteredName", None)]),
    ],
    ids=["bare", "long-view", "two-bare", "three-mixed", "whitespace", "plus-inside-brackets", "unregistered"],
)
def test_featurizer_atoms_are_parsed(recipe: str, expected: list[tuple[str, str | None]]) -> None:
    assert parse_featurizer_atoms(recipe) == expected


@pytest.mark.parametrize(
    "recipe", ["a+", "+a", "a++b", "a+[", "]+a", "view[a:b]", "[x]", "a[b][c]", "", "   ", "view[]"]
)
def test_featurizer_recipe_rejects_malformed_shapes(recipe: str) -> None:
    with pytest.raises(ValueError, match="Malformed featurizer recipe"):
        parse_featurizer_atoms(recipe)


def test_featurizer_error_mentions_non_empty_atoms() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        parse_featurizer_atoms("scaledGeneExpression+")


@pytest.mark.parametrize(
    ("recipe", "expected"),
    [
        ("scaledGeneExpression", {"name": "scaledGeneExpression"}),
        ("view[expression]", {"name": "view", "view": "expression"}),
        (
            "view[expression]+pca[methylation]",
            {
                "name": "concatFeaturizers",
                "featurizers": [{"name": "view", "view": "expression"}, {"name": "pca", "view": "methylation"}],
            },
        ),
        ("view[a+b]", {"name": "view", "view": "a+b"}),
        # Expansion transcribes without judging: unknown names and views pass through.
        ("scaledGeneExpression[gene_expression]", {"name": "scaledGeneExpression", "view": "gene_expression"}),
        ("notARegisteredName", {"name": "notARegisteredName"}),
    ],
    ids=["bare", "bracket", "two-bracket", "plus-inside-brackets", "bracket-on-non-view", "unknown-name"],
)
def test_recipe_expands_to_the_mapping_yaml_would_spell_out(recipe: str, expected: dict) -> None:
    assert expand_featurizer_recipe(recipe) == expected


def test_expansion_rejects_a_blank_token() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        expand_featurizer_recipe("   ")


def _slots(spec: str) -> tuple[str | None, str | None, str]:
    payload = parse_model_recipe(spec)
    sample, drug = payload["sample_featurizer"], payload["drug_featurizer"]
    return (
        sample["name"] if sample is not None else None,
        drug["name"] if drug is not None else None,
        payload["predictor"],
    )


def test_model_recipe_payload_carries_exactly_the_config_field_keys() -> None:
    assert parse_model_recipe("view[expression]+landmarkGenes:fingerprints:randomForest") == {
        "sample_featurizer": {
            "name": "concatFeaturizers",
            "featurizers": [{"name": "view", "view": "expression"}, {"name": "landmarkGenes"}],
        },
        "drug_featurizer": {"name": "fingerprints"},
        "predictor": "randomForest",
    }
    payload = parse_model_recipe("view[expression]:view[expression]:randomForest")
    assert payload["sample_featurizer"] == payload["drug_featurizer"] == {"name": "view", "view": "expression"}


def test_two_part_model_recipe_leaves_the_drug_slot_unset() -> None:
    assert parse_model_recipe("scaledGeneExpression:singleDrugElasticNet") == {
        "sample_featurizer": {"name": "scaledGeneExpression"},
        "drug_featurizer": None,
        "predictor": "singleDrugElasticNet",
    }


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("elasticNet", (None, None, "elasticNet")),
        ("scaledGeneExpression:fingerprints:elasticNet", ("scaledGeneExpression", "fingerprints", "elasticNet")),
        (
            "view[expression]+view[mutations]:fingerprints:randomForest",
            ("concatFeaturizers", "fingerprints", "randomForest"),
        ),
        (
            "  scaledGeneExpression : fingerprints : elasticNet  ",
            ("scaledGeneExpression", "fingerprints", "elasticNet"),
        ),
    ],
    ids=["predictor-only", "three-slot", "concat-slot", "whitespace"],
)
def test_model_recipe_slots_are_split(spec: str, expected: tuple[str | None, str | None, str]) -> None:
    assert _slots(spec) == expected


@pytest.mark.parametrize(
    "spec", ["a:b:c:d", " :b:c", ":x", "a::b", "a:", "view[a:b]:x:y", "view[a:b]:fingerprints:randomForest"]
)
def test_model_recipe_rejects_malformed_specs(spec: str) -> None:
    with pytest.raises(ValueError, match="Malformed model recipe"):
        parse_model_recipe(spec)


@pytest.mark.parametrize("spec", ["", "   "])
def test_model_recipe_rejects_blank_specs(spec: str) -> None:
    with pytest.raises(ValueError, match="must be a non-empty string"):
        parse_model_recipe(spec)


@pytest.mark.parametrize(
    ("slots", "expected"),
    [
        ((None, None, "naiveMean"), "naiveMean"),
        (("scaledGeneExpression", None, "singleDrugElasticNet"), "scaledGeneExpression:singleDrugElasticNet"),
        (("scaledGeneExpression", "fingerprints", "elasticNet"), "scaledGeneExpression:fingerprints:elasticNet"),
        (
            ("view[expression]+view[mutations]", "fingerprints", "randomForest"),
            "view[expression]+view[mutations]:fingerprints:randomForest",
        ),
    ],
    ids=["predictor-only", "two-slot", "three-slot", "concat-slot"],
)
def test_model_recipe_is_formatted_from_slots(slots: tuple[str | None, str | None, str], expected: str) -> None:
    assert format_model_recipe(*slots) == expected
    if "+" not in expected:
        assert _slots(expected) == slots


def test_formatting_rejects_missing_slots() -> None:
    with pytest.raises(ValueError, match="predictor is required"):
        format_model_recipe("scaledGeneExpression", "fingerprints", "")
    with pytest.raises(ValueError, match="sample is required when drug is set"):
        format_model_recipe(None, "fingerprints", "elasticNet")


def _space(cls: type, **defaults: Any) -> dict[str, Any]:
    """The class's own space with some defaults moved, as ``defaults:`` records it."""
    space = cls.get_hyperparameter_space()
    return {key: {**spec, "default": defaults[key]} if key in defaults else spec for key, spec in space.items()}


def _side_of(raw: object) -> str:
    """The side a test featurizer config is used on: the drug side for ``fingerprints``."""
    return "drug" if "fingerprints" in repr(raw) else "sample"


def _sided(fields: dict, side: str) -> dict:
    """*fields* with ``side`` set on the node and its concat children, as normalizing writes them."""
    sided = {**fields, "side": side}
    if "featurizers" in fields:
        sided["featurizers"] = [_sided(child, side) for child in fields["featurizers"]]
    return sided


_CANONICAL_CONCAT = {
    "name": "concatFeaturizers",
    "featurizers": [{"name": "view", "view": "gene_expression"}, {"name": "view", "view": "mutations"}],
}
_PCA_SPACE = {"n_components": {"type": "int", "low": 2, "high": 99, "default": 5}}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("scaledGeneExpression", {"name": "scaledGeneExpression"}),
        ("view[gene_expression]", {"name": "view", "view": "gene_expression"}),
        ("view[gene_expression]+view[mutations]", _CANONICAL_CONCAT),
        ({"name": "view", "view": "gene_expression"}, {"name": "view", "view": "gene_expression"}),
        (_CANONICAL_CONCAT, _CANONICAL_CONCAT),
        (
            {
                "name": "concatFeaturizers",
                "featurizers": ["view[gene_expression]", {"name": "view", "view": "mutations"}],
            },
            _CANONICAL_CONCAT,
        ),
        (
            {"name": "scaledGeneExpression", "options": {"gene_list": "landmark_genes"}},
            {"name": "scaledGeneExpression", "options": {"gene_list": "landmark_genes"}},
        ),
        (
            {"name": "fingerprints", "defaults": {"n_bits": 512}},
            {
                "name": "fingerprints",
                "hyperparameter_space": _space(get_featurizer("fingerprints", "drug"), n_bits=512),
            },
        ),
        (
            {"name": "pca", "view": "methylation", "hyperparameter_space": _PCA_SPACE, "defaults": {"n_components": 8}},
            {
                "name": "pca",
                "view": "methylation",
                "hyperparameter_space": {"n_components": {**_PCA_SPACE["n_components"], "default": 8}},
            },
        ),
        (
            {"name": "landmarkGenes", "options": {"arcsinh": False}, "defaults": {"standardize": False}},
            {
                "name": "landmarkGenes",
                "options": {"arcsinh": False},
                "hyperparameter_space": _space(get_featurizer("landmarkGenes", "sample"), standardize=False),
            },
        ),
    ],
    ids=[
        "name",
        "name-view",
        "plus-recipe",
        "canonical",
        "canonical-concat",
        "concat-children-as-recipes",
        "options",
        "defaults",
        "defaults-in-declared-space",
        "options-and-defaults",
    ],
)
def test_accepted_featurizer_spellings_normalize_to_canonical_fields(raw: object, expected: dict) -> None:
    side = _side_of(raw)
    assert normalize_featurizer(raw, side) == _sided(expected, side)
    assert FeaturizerConfig.parse(raw, side) == FeaturizerConfig.parse(expected, side)


def test_normalizing_needs_a_side_and_keeps_a_matching_one() -> None:
    with pytest.raises(ValueError, match="featurizer 'pca' needs a side"):
        normalize_featurizer({"name": "pca", "view": "methylation"})
    assert normalize_featurizer({"name": "pca", "view": "methylation", "side": "sample"})["side"] == "sample"
    with pytest.raises(ValueError, match="featurizer 'fingerprints' has side 'drug' but is used on the sample side"):
        normalize_featurizer({"name": "fingerprints", "side": "drug"}, "sample")


@pytest.mark.parametrize(
    ("raw", "error", "match"),
    [
        (
            ["view[gene_expression]", "view[mutations]"],
            ValueError,
            "featurizer list is not accepted; write 'view[gene_expression]+view[mutations]'",
        ),
        (
            ["oneHot", {"name": "tissue"}],
            ValueError,
            "write {'name': 'concatFeaturizers', 'featurizers': ['oneHot', {'name': 'tissue'}]}",
        ),
        (
            {"fingerprints": {"n_bits": 512}},
            ValueError,
            "one-key featurizer mapping {'fingerprints': ...} is not accepted; "
            "write {'name': 'fingerprints', 'options': {'n_bits': 512}}, "
            "with tunable hyperparameters under 'defaults' instead of 'options'",
        ),
        (
            {"pca[methylation]": None},
            ValueError,
            "write {'name': 'pca', 'view': 'methylation'}",
        ),
        (
            {"concatFeaturizers": {"featurizers": ["scaledGeneExpression", "view[mutations]"]}},
            ValueError,
            "write {'name': 'concatFeaturizers', 'featurizers': ['scaledGeneExpression', 'view[mutations]']}",
        ),
        ({"view": "methylation", "options": {}}, ValueError, "needs a 'name' key, got keys ['options', 'view']"),
        (
            {"name": "fingerprints", "defaults": {"not_a_knob": 1}},
            ValueError,
            "defaults of drug featurizer 'fingerprints': 'not_a_knob' is not a tunable hyperparameter "
            "(tunable: n_bits, radius, use_chirality, use_counts); put fixed constructor values under 'options'",
        ),
        (
            {"name": "fingerprints", "defaults": 5},
            ValueError,
            "defaults of drug featurizer 'fingerprints' must be a mapping",
        ),
        (
            {"name": "pca", "view": "methylation", "options": {"n_components": 4}},
            ValueError,
            "options of sample featurizer 'pca' name tunable hyperparameters (n_components); "
            "set their value under 'defaults' instead",
        ),
        (
            {"name": "concatFeaturizers", "featurizers": "scaledGeneExpression"},
            ValueError,
            "featurizers must be a list when set",
        ),
        (123, TypeError, "recipe string or a mapping with 'name'"),
    ],
    ids=[
        "list-of-recipes",
        "list-of-mappings",
        "one-key",
        "one-key-without-body",
        "one-key-concat",
        "no-name",
        "defaults-not-tunable",
        "defaults-not-a-mapping",
        "tunable-option",
        "children-not-a-list",
        "not-a-config",
    ],
)
def test_rejected_featurizer_spellings_name_the_canonical_form(raw: object, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=re.escape(match)):
        normalize_featurizer(raw, _side_of(raw))


@pytest.mark.parametrize(
    ("name", "view"),
    [("view", "expression"), ("view", "custom_test_view"), ("fingerprints", "ignored")],
    ids=["canonical", "custom-view", "non-view-parametric"],
)
def test_views_are_passed_through_unchanged(name, view) -> None:
    side = _side_of(name)
    payload = normalize_featurizer({"name": name, "view": view}, side)
    assert payload == {"name": name, "view": view, "side": side}
    assert normalize_featurizer(f"{name}[{view}]", side) == payload


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("view", "requires an explicit view"),
        ("pca", "requires an explicit view"),
        ({"name": "pca"}, "requires an explicit view"),
        ({"name": "pca", "view": "  "}, "view must be a non-empty string"),
    ],
    ids=["view", "pca", "missing", "blank"],
)
def test_view_parametric_featurizer_without_a_usable_view_is_rejected(raw, match) -> None:
    with pytest.raises(ValueError, match=match):
        FeaturizerConfig.parse(raw, "sample")


PREDICTOR_SPACE = {"n_estimators": {"type": "int", "low": 2, "high": 20, "default": 4}}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("randomForest", {"name": "randomForest"}),
        ({"name": "randomForest"}, {"name": "randomForest"}),
        (
            {"name": "randomForest", "hyperparameter_space": PREDICTOR_SPACE},
            {"name": "randomForest", "hyperparameter_space": PREDICTOR_SPACE},
        ),
        (
            {"name": "randomForest", "defaults": {"n_estimators": 10}},
            {"name": "randomForest", "hyperparameter_space": _space(get_predictor("randomForest"), n_estimators=10)},
        ),
        (
            {"name": "randomForest", "hyperparameter_space": PREDICTOR_SPACE, "defaults": {"n_estimators": 10}},
            {
                "name": "randomForest",
                "hyperparameter_space": {"n_estimators": {**PREDICTOR_SPACE["n_estimators"], "default": 10}},
            },
        ),
    ],
    ids=["name", "canonical", "space", "defaults", "defaults-in-declared-space"],
)
def test_accepted_predictor_spellings_normalize_to_canonical_fields(raw: object, expected: dict) -> None:
    assert normalize_predictor(raw) == expected


@pytest.mark.parametrize(
    ("raw", "error", "match"),
    [
        (
            {"neuralNetwork": {"units_per_layer": [16, 8, 4]}},
            ValueError,
            "the one-key form is not accepted, "
            "write {'name': 'neuralNetwork', 'defaults': {'units_per_layer': [16, 8, 4]}}",
        ),
        ({"randomForest": None}, ValueError, "write {'name': 'randomForest'}"),
        ({"hyperparameter_space": {}, "extra": 1}, ValueError, "A predictor mapping needs a 'name' key"),
        (
            {"name": "randomForest", "defaults": {"not_a_knob": 1}},
            ValueError,
            "defaults of predictor 'randomForest': 'not_a_knob' is not a tunable hyperparameter",
        ),
        (123, TypeError, "name or a mapping with 'name'"),
    ],
    ids=["one-key", "one-key-without-body", "no-name", "defaults-not-tunable", "not-a-config"],
)
def test_rejected_predictor_spellings_name_the_canonical_form(raw: object, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=re.escape(match)):
        normalize_predictor(raw)


def test_defaults_do_not_mutate_the_class_space() -> None:
    original = get_featurizer("fingerprints", "drug").get_hyperparameter_space()
    normalize_featurizer({"name": "fingerprints", "defaults": {"n_bits": 512}}, "drug")
    assert get_featurizer("fingerprints", "drug").get_hyperparameter_space() == original


def test_the_suggested_form_of_a_nested_removed_spelling_is_canonical_all_the_way_down() -> None:
    with pytest.raises(ValueError, match="A featurizer list is not accepted") as info:
        normalize_featurizer(["oneHot", {"tissue": {"allow_missing": True}}], "sample")
    hint = str(info.value).split("; write ", 1)[1]
    assert hint == repr(
        {
            "name": "concatFeaturizers",
            "featurizers": ["oneHot", {"name": "tissue", "options": {"allow_missing": True}}],
        }
    )
    assert normalize_featurizer(ast.literal_eval(hint), "sample")["featurizers"][1] == {
        "name": "tissue",
        "side": "sample",
        "options": {"allow_missing": True},
    }
