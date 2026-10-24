"""Tests for the recipe language: the parsing grammar, expansion, and the formatter (no registry needed)."""

from __future__ import annotations

import pytest

from drevalpy.models.config._recipe import (
    expand_featurizer_recipe,
    format_model_recipe,
    parse_featurizer_atoms,
    parse_model_recipe,
)


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
