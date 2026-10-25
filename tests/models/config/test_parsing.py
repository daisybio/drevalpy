"""Parsing of model configs: the recipe grammar, featurizer/predictor normalization and space defaults."""

from __future__ import annotations

from typing import Any

import pytest

from drevalpy.models.config._featurizer_parse import normalize_featurizer_config
from drevalpy.models.config._predictor_parse import normalize_predictor_config
from drevalpy.models.config._recipe import (
    expand_featurizer_recipe,
    format_model_recipe,
    parse_featurizer_atoms,
    parse_model_recipe,
)
from drevalpy.models.config._space_defaults import split_space_and_options


@pytest.mark.parametrize(
    ("recipe", "expected"),
    [
        ("scaledGeneExpression", [("scaledGeneExpression", None)]),
        ("raw[copy_number_variation_gistic]", [("raw", "copy_number_variation_gistic")]),
        ("fingerprints+identity", [("fingerprints", None), ("identity", None)]),
        (
            "raw[expression]+pca[proteomics]+scaledGeneExpression",
            [("raw", "expression"), ("pca", "proteomics"), ("scaledGeneExpression", None)],
        ),
        ("  raw[expression] + raw[mutations]  ", [("raw", "expression"), ("raw", "mutations")]),
        ("raw[a+b]", [("raw", "a+b")]),
        ("notARegisteredName", [("notARegisteredName", None)]),
    ],
    ids=["bare", "long-view", "two-bare", "three-mixed", "whitespace", "plus-inside-brackets", "unregistered"],
)
def test_featurizer_atoms_are_parsed(recipe: str, expected: list[tuple[str, str | None]]) -> None:
    assert parse_featurizer_atoms(recipe) == expected


@pytest.mark.parametrize("recipe", ["a+", "+a", "a++b", "a+[", "]+a", "raw[a:b]", "[x]", "a[b][c]", "", "   ", "raw[]"])
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
        ("raw[expression]", {"name": "raw", "view": "expression"}),
        (
            "raw[expression]+pca[methylation]",
            {
                "name": "concatFeaturizers",
                "featurizers": [{"name": "raw", "view": "expression"}, {"name": "pca", "view": "methylation"}],
            },
        ),
        ("raw[a+b]", {"name": "raw", "view": "a+b"}),
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
    assert parse_model_recipe("raw[expression]+landmarkGenes:fingerprints:randomForest") == {
        "sample_featurizer": {
            "name": "concatFeaturizers",
            "featurizers": [{"name": "raw", "view": "expression"}, {"name": "landmarkGenes"}],
        },
        "drug_featurizer": {"name": "fingerprints"},
        "predictor": "randomForest",
    }
    payload = parse_model_recipe("raw[expression]:raw[expression]:randomForest")
    assert payload["sample_featurizer"] == payload["drug_featurizer"] == {"name": "raw", "view": "expression"}


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
            "raw[expression]+raw[mutations]:fingerprints:randomForest",
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
    "spec", ["a:b:c:d", " :b:c", ":x", "a::b", "a:", "raw[a:b]:x:y", "raw[a:b]:fingerprints:randomForest"]
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
            ("raw[expression]+raw[mutations]", "fingerprints", "randomForest"),
            "raw[expression]+raw[mutations]:fingerprints:randomForest",
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


def _normalize_recipe(recipe: str, *, default_registry: str) -> dict:
    return normalize_featurizer_config(expand_featurizer_recipe(recipe), default_registry=default_registry)


def test_normalize_named_mapping() -> None:
    payload = normalize_featurizer_config({"name": "fingerprints"}, default_registry="drug")
    assert payload == {"name": "fingerprints", "registry": "drug"}


@pytest.mark.parametrize("value", ["fingerprints", 123])
def test_normalize_rejects_invalid_shape(value) -> None:
    with pytest.raises(TypeError, match="list or mapping"):
        normalize_featurizer_config(value, default_registry="drug")


def test_normalize_list_shorthand() -> None:
    payload = normalize_featurizer_config(
        ["scaledGeneExpression", "raw[mutations]", {"pca[methylation]": {"n_components": 64}}],
        default_registry="sample",
    )
    assert payload["name"] == "concatFeaturizers"
    assert payload["registry"] == "sample"
    children = payload["featurizers"]
    assert children[0]["name"] == "scaledGeneExpression"
    assert (children[1]["name"], children[1]["view"]) == ("raw", "mutations")
    assert (children[2]["name"], children[2]["view"]) == ("pca", "methylation")
    assert children[2]["hyperparameter_space"]["n_components"]["default"] == 64
    assert all(child["registry"] == "sample" for child in children)


def test_normalize_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalize_featurizer_config([], default_registry="sample")


def test_a_recipe_and_the_equivalent_yaml_normalize_alike() -> None:
    from_recipe = _normalize_recipe("raw[expression]+pca[methylation]", default_registry="sample")
    from_yaml = normalize_featurizer_config(
        {
            "name": "concatFeaturizers",
            "featurizers": [{"name": "raw", "view": "expression"}, {"name": "pca", "view": "methylation"}],
        },
        default_registry="sample",
    )
    assert from_recipe == from_yaml
    assert [child["view"] for child in from_recipe["featurizers"]] == ["expression", "methylation"]


def test_normalize_records_the_registry_on_every_child() -> None:
    children = _normalize_recipe("fingerprints+identity", default_registry="drug")["featurizers"]
    assert [child["name"] for child in children] == ["fingerprints", "identity"]
    assert all(child["registry"] == "drug" for child in children)


@pytest.mark.parametrize(
    ("name", "view", "registry"),
    [("raw", "expression", "sample"), ("raw", "custom_test_view", "sample"), ("fingerprints", "ignored", "drug")],
    ids=["canonical", "custom-view", "non-view-parametric"],
)
def test_views_are_passed_through_unchanged(name, view, registry) -> None:
    payload = normalize_featurizer_config({"name": name, "view": view}, default_registry=registry)
    assert payload == {"name": name, "view": view, "registry": registry}
    if registry == "sample":
        assert _normalize_recipe(f"{name}[{view}]", default_registry=registry) == payload


@pytest.mark.parametrize("recipe", ["raw", "pca"])
def test_normalize_rejects_bare_raw_or_pca(recipe) -> None:
    with pytest.raises(ValueError, match="requires an explicit view"):
        _normalize_recipe(recipe, default_registry="sample")


@pytest.mark.parametrize("mapping", [{"name": "pca"}, {"name": "pca", "view": "  "}], ids=["missing", "blank"])
def test_named_mapping_without_a_usable_view_is_rejected(mapping) -> None:
    with pytest.raises(ValueError, match="requires an explicit view"):
        normalize_featurizer_config(mapping, default_registry="sample")


_EXPLICIT_SPACE = {"n_components": {"type": "int", "low": 2, "high": 99, "default": 5}}


@pytest.mark.parametrize(
    ("body", "expected_space_default", "expected_options"),
    [
        ({"n_components": 8}, 8, None),
        ({"hyperparameter_space": _EXPLICIT_SPACE, "n_components": 8}, 5, None),
        ({"options": {"foo": 1}, "bar": 2}, None, {"foo": 1, "bar": 2}),
    ],
    ids=["simple-value", "explicit-space-wins", "options-and-simple"],
)
def test_one_key_body_folds_loose_values(body: dict, expected_space_default, expected_options) -> None:
    payload = normalize_featurizer_config({"pca[methylation]": body}, default_registry="sample")
    assert (payload["name"], payload["view"]) == ("pca", "methylation")
    if expected_space_default is not None:
        assert payload["hyperparameter_space"]["n_components"]["default"] == expected_space_default
    if expected_options is not None:
        assert payload["options"] == expected_options


@pytest.mark.parametrize("key", ["a+b", "raw["])
def test_unparsable_one_key_falls_back_to_the_registry_error(key) -> None:
    with pytest.raises(ValueError, match="Unknown Sample featurizer"):
        normalize_featurizer_config({key: {"foo": 1}}, default_registry="sample")


def test_one_key_loose_values_resolve_against_the_drug_registry() -> None:
    payload = normalize_featurizer_config({"fingerprints": {"n_bits": 512}}, default_registry="drug")
    assert payload["registry"] == "drug"
    assert payload["hyperparameter_space"]["n_bits"]["default"] == 512


def test_one_key_with_a_bracket_records_the_view_without_judging_it() -> None:
    payload = normalize_featurizer_config(
        {"scaledGeneExpression[gene_expression]": {"foo": 1}}, default_registry="sample"
    )
    assert (payload["name"], payload["view"], payload["options"]) == (
        "scaledGeneExpression",
        "gene_expression",
        {"foo": 1},
    )


def test_one_key_body_shapes() -> None:
    assert normalize_featurizer_config({"scaledGeneExpression": None}, default_registry="sample") == {
        "name": "scaledGeneExpression",
        "registry": "sample",
    }
    payload = normalize_featurizer_config(
        {"concatFeaturizers": {"featurizers": ["scaledGeneExpression", "raw[mutations]"]}}, default_registry="sample"
    )
    assert [child["name"] for child in payload["featurizers"]] == ["scaledGeneExpression", "raw"]
    with pytest.raises(ValueError, match="must be a mapping when provided"):
        normalize_featurizer_config({"scaledGeneExpression": 5}, default_registry="sample")


def test_malformed_mappings_are_rejected() -> None:
    with pytest.raises(ValueError, match="featurizers must be a list when set"):
        normalize_featurizer_config(
            {"name": "concatFeaturizers", "featurizers": "scaledGeneExpression"}, default_registry="sample"
        )
    with pytest.raises(ValueError, match="list, one-key mapping, or dict with 'name'"):
        normalize_featurizer_config({"view": "methylation", "options": {}}, default_registry="sample")


PREDICTOR_SPACE = {"n_estimators": {"type": "int", "low": 2, "high": 20, "default": 4}}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("randomForest", {"name": "randomForest"}),
        ({"randomForest": None}, {"name": "randomForest"}),
        (
            {"name": "randomForest", "hyperparameter_space": PREDICTOR_SPACE},
            {"name": "randomForest", "hyperparameter_space": PREDICTOR_SPACE},
        ),
    ],
)
def test_normalize_shapes(raw: object, expected: dict) -> None:
    assert normalize_predictor_config(raw) == expected


@pytest.mark.parametrize(
    ("raw", "default"),
    [
        ({"randomForest": {"n_estimators": 10}}, 10),
        ({"randomForest": {"hyperparameter_space": PREDICTOR_SPACE, "n_estimators": 10}}, 4),
    ],
)
def test_loose_value_becomes_default_unless_space_given(raw: dict, default: int) -> None:
    payload = normalize_predictor_config(raw)
    assert payload["hyperparameter_space"]["n_estimators"]["default"] == default


@pytest.mark.parametrize(
    ("raw", "error", "match"),
    [
        ({"randomForest": {"not_a_knob": 1}}, ValueError, "do not accept non-tunable options \\('not_a_knob'\\)"),
        ({"randomForest": 5}, ValueError, "must be a mapping when provided"),
        (123, TypeError, "string or mapping"),
        ({"hyperparameter_space": {}, "extra": 1}, ValueError, "string, one-key mapping, or dict with 'name'"),
    ],
)
def test_invalid_shapes_rejected(raw: object, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        normalize_predictor_config(raw)


class _Component:
    @staticmethod
    def get_hyperparameter_space() -> dict[str, Any]:
        return {"alpha": {"type": "float", "low": 0.0, "high": 1.0, "default": 0.5}, "not_a_spec": "opaque"}


@pytest.mark.parametrize(
    ("values", "alpha_default", "options"),
    [
        ({"alpha": 0.9}, 0.9, {}),
        ({"device": "cpu"}, 0.5, {"device": "cpu"}),
        ({"not_a_spec": 3}, 0.5, {"not_a_spec": 3}),
        ({}, 0.5, {}),
    ],
)
def test_values_move_defaults_or_become_options(values: dict, alpha_default: float, options: dict) -> None:
    space, got_options = split_space_and_options(_Component, values)
    assert set(space) == {"alpha", "not_a_spec"}
    assert space["alpha"]["default"] == alpha_default
    assert got_options == options


def test_the_declared_space_is_not_mutated() -> None:
    original = _Component.get_hyperparameter_space()
    split_space_and_options(_Component, {"alpha": 0.9})
    assert _Component.get_hyperparameter_space() == original
