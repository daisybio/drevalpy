"""Tests for the shared recipe-string grammar."""

from __future__ import annotations

import pytest

from drevalpy.models.config._recipe import parse_featurizer_atoms, parse_model_recipe


@pytest.mark.parametrize(
    ("recipe", "expected"),
    [
        ("scaledGeneExpression", [("scaledGeneExpression", None)]),
        ("raw[expression]", [("raw", "expression")]),
        ("raw[copy_number_variation_gistic]", [("raw", "copy_number_variation_gistic")]),
        ("fingerprints+identity", [("fingerprints", None), ("identity", None)]),
        ("raw[expression]+raw[mutations]", [("raw", "expression"), ("raw", "mutations")]),
        (
            "raw[expression]+pca[proteomics]+scaledGeneExpression",
            [("raw", "expression"), ("pca", "proteomics"), ("scaledGeneExpression", None)],
        ),
        ("  raw[expression] + raw[mutations]  ", [("raw", "expression"), ("raw", "mutations")]),
    ],
    ids=["bare", "bracket", "long-view", "two-bare", "two-bracket", "three-mixed", "whitespace"],
)
def test_featurizer_atoms_are_parsed(recipe: str, expected: list[tuple[str, str | None]]) -> None:
    """Every valid featurizer notation resolves to its ``(name, view)`` atoms.

    :param recipe: Recipe string to parse.
    :param expected: Expected ``(name, view)`` pairs.
    """
    assert parse_featurizer_atoms(recipe) == expected


def test_plus_inside_brackets_stays_one_atom() -> None:
    """A ``+`` within a view must not split the recipe, so the bad view is named in the error."""
    assert parse_featurizer_atoms("raw[a+b]") == [("raw", "a+b")]


@pytest.mark.parametrize(
    "recipe",
    ["a+", "+a", "a++b", "a+[", "]+a", "raw[a:b]", "[x]", "a[b][c]", "", "   ", "raw[]"],
)
def test_featurizer_recipe_rejects_malformed_shapes(recipe: str) -> None:
    """Shapes that are not ``name`` or ``name[view]`` joined by ``+`` are rejected up front.

    :param recipe: Malformed recipe string.
    """
    with pytest.raises(ValueError, match="Malformed featurizer recipe"):
        parse_featurizer_atoms(recipe)


def test_featurizer_error_mentions_non_empty_atoms() -> None:
    """The message keeps the wording that callers and older tests match on."""
    with pytest.raises(ValueError, match="non-empty"):
        parse_featurizer_atoms("scaledGeneExpression+")


def test_unregistered_names_are_left_to_the_registry() -> None:
    """Shape-valid names pass through; existence is the registry's question, not the grammar's."""
    assert parse_featurizer_atoms("notARegisteredName") == [("notARegisteredName", None)]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("elasticNet", (None, None, "elasticNet")),
        ("scaledGeneExpression:singleDrugElasticNet", ("scaledGeneExpression", None, "singleDrugElasticNet")),
        (
            "scaledGeneExpression:fingerprints:elasticNet",
            ("scaledGeneExpression", "fingerprints", "elasticNet"),
        ),
        (
            "raw[expression]+raw[mutations]:fingerprints:randomForest",
            ("raw[expression]+raw[mutations]", "fingerprints", "randomForest"),
        ),
        (
            "  scaledGeneExpression : fingerprints : elasticNet  ",
            ("scaledGeneExpression", "fingerprints", "elasticNet"),
        ),
    ],
    ids=["predictor-only", "two-slot", "three-slot", "concat-slot", "whitespace"],
)
def test_model_recipe_slots_are_split(spec: str, expected: tuple[str | None, str | None, str]) -> None:
    """Featurizer slots come back as unparsed recipe strings, stripped of padding.

    :param spec: Model recipe string.
    :param expected: Expected ``(cell_line, drug, predictor)`` tuple.
    """
    assert parse_model_recipe(spec) == expected


@pytest.mark.parametrize("spec", ["", "   ", "a:b:c:d", " :b:c", ":x", "a::b", "a:", "raw[a:b]:x:y"])
def test_model_recipe_rejects_malformed_specs(spec: str) -> None:
    """Wrong slot counts and empty slots are rejected by the grammar itself.

    :param spec: Malformed model recipe string.
    """
    with pytest.raises(ValueError, match="Malformed model recipe"):
        parse_model_recipe(spec)


def test_colon_inside_a_view_is_not_a_slot_separator() -> None:
    """The old ``str.split(':')`` was bracket-unaware; the grammar is not."""
    with pytest.raises(ValueError, match="Malformed model recipe"):
        parse_model_recipe("raw[a:b]:fingerprints:randomForest")
