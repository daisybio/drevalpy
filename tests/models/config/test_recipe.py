"""Tests for the recipe language: the parsing grammar, expansion, and the formatter.

Expansion is purely structural, so nothing here needs the component registry.
"""

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
    ("recipe", "expected"),
    [
        ("scaledGeneExpression", {"name": "scaledGeneExpression"}),
        ("raw[expression]", {"name": "raw", "view": "expression"}),
        (
            "landmarkGenes+normalizedProteomics",
            {
                "name": "concatFeaturizers",
                "featurizers": [{"name": "landmarkGenes"}, {"name": "normalizedProteomics"}],
            },
        ),
        (
            "raw[expression]+pca[methylation]",
            {
                "name": "concatFeaturizers",
                "featurizers": [
                    {"name": "raw", "view": "expression"},
                    {"name": "pca", "view": "methylation"},
                ],
            },
        ),
    ],
    ids=["bare", "bracket", "two-bare", "two-bracket"],
)
def test_recipe_expands_to_the_mapping_yaml_would_spell_out(recipe: str, expected: dict) -> None:
    """A recipe is shorthand for a field mapping, so expansion yields exactly that mapping.

    :param recipe: Recipe string to expand.
    :param expected: Expected field mapping.
    """
    assert expand_featurizer_recipe(recipe) == expected


def test_expansion_rejects_a_blank_token() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        expand_featurizer_recipe("   ")


@pytest.mark.parametrize(
    "recipe",
    [
        "scaledGeneExpression[gene_expression]",
        "raw[not_a_view]",
        "raw[custom_matrix]",
        "notARegisteredName",
    ],
    ids=["bracket-on-non-view-featurizer", "unknown-view", "custom-view", "unknown-name"],
)
def test_expansion_asks_no_semantic_questions(recipe: str) -> None:
    """Expansion transcribes; it does not judge.

    Writing a view in brackets says no more than writing one as a ``view`` key, so neither is
    checked here. Whether the featurizer exists, takes a view, or can supply that view is
    settled downstream, identically for both notations.

    :param recipe: Recipe whose meaning is questionable but whose shape is fine.
    """
    payload = expand_featurizer_recipe(recipe)
    assert payload["name"]


def test_expansion_keeps_the_view_as_written() -> None:
    """Resolving is normalization's job, so a bracket and a spelled-out view arrive alike."""
    assert expand_featurizer_recipe("raw[expression]")["view"] == "expression"


def test_a_plus_inside_brackets_stays_in_its_atom() -> None:
    """The view keeps the ``+``, so a later error names the view, not a truncated featurizer."""
    assert expand_featurizer_recipe("raw[a+b]") == {"name": "raw", "view": "a+b"}


def _slots(spec: str) -> tuple[str | None, str | None, str]:
    """Read a recipe's payload back as the triple the formatter takes.

    Each featurizer slot is expanded into a mapping, so the slot's name stands in for it; a
    concat slot is named after the node it expands to.

    :param spec: Model recipe string.
    :returns: Cell-line slot name, drug slot name, and predictor name.
    """
    payload = parse_model_recipe(spec)
    cell = payload["cell_line_featurizer"]
    drug = payload["drug_featurizer"]
    return (
        cell["name"] if cell is not None else None,
        drug["name"] if drug is not None else None,
        payload["predictor"],
    )


def test_model_recipe_payload_carries_exactly_the_config_field_keys() -> None:
    """The mapping goes straight into ``from_dict``, so it names config fields and nothing else."""
    assert parse_model_recipe("raw[expression]+landmarkGenes:fingerprints:randomForest") == {
        "cell_line_featurizer": {
            "name": "concatFeaturizers",
            "featurizers": [
                {"name": "raw", "view": "expression"},
                {"name": "landmarkGenes"},
            ],
        },
        "drug_featurizer": {"name": "fingerprints"},
        "predictor": "randomForest",
    }


def test_both_slots_are_expanded_the_same_way() -> None:
    """No registry is involved, so a slot is transcribed the same wherever it appears."""
    payload = parse_model_recipe("raw[expression]:raw[expression]:randomForest")
    assert payload["cell_line_featurizer"] == payload["drug_featurizer"] == {"name": "raw", "view": "expression"}


def test_two_part_model_recipe_leaves_the_drug_slot_unset() -> None:
    """``ModelConfig`` injects the routing featurizer, so the payload states no drug slot."""
    assert parse_model_recipe("scaledGeneExpression:singleDrugElasticNet") == {
        "cell_line_featurizer": {"name": "scaledGeneExpression"},
        "drug_featurizer": None,
        "predictor": "singleDrugElasticNet",
    }


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
            ("concatFeaturizers", "fingerprints", "randomForest"),
        ),
        (
            "  scaledGeneExpression : fingerprints : elasticNet  ",
            ("scaledGeneExpression", "fingerprints", "elasticNet"),
        ),
    ],
    ids=["predictor-only", "two-slot", "three-slot", "concat-slot", "whitespace"],
)
def test_model_recipe_slots_are_split(spec: str, expected: tuple[str | None, str | None, str]) -> None:
    """Each slot becomes its own featurizer mapping, stripped of padding.

    :param spec: Model recipe string.
    :param expected: Expected ``(cell_line, drug, predictor)`` names.
    """
    assert _slots(spec) == expected


@pytest.mark.parametrize("spec", ["a:b:c:d", " :b:c", ":x", "a::b", "a:", "raw[a:b]:x:y"])
def test_model_recipe_rejects_malformed_specs(spec: str) -> None:
    """Wrong slot counts and empty slots are rejected by the grammar itself.

    :param spec: Malformed model recipe string.
    """
    with pytest.raises(ValueError, match="Malformed model recipe"):
        parse_model_recipe(spec)


@pytest.mark.parametrize("spec", ["", "   "])
def test_model_recipe_rejects_blank_specs(spec: str) -> None:
    """A blank recipe is reported as empty rather than as a grammar failure.

    :param spec: Blank model recipe string.
    """
    with pytest.raises(ValueError, match="must be a non-empty string"):
        parse_model_recipe(spec)


def test_colon_inside_a_view_is_not_a_slot_separator() -> None:
    """A colon inside brackets is not treated as a slot separator."""
    with pytest.raises(ValueError, match="Malformed model recipe"):
        parse_model_recipe("raw[a:b]:fingerprints:randomForest")


@pytest.mark.parametrize(
    ("slots", "expected"),
    [
        ((None, None, "naiveMean"), "naiveMean"),
        (("scaledGeneExpression", None, "singleDrugElasticNet"), "scaledGeneExpression:singleDrugElasticNet"),
        (
            ("scaledGeneExpression", "fingerprints", "elasticNet"),
            "scaledGeneExpression:fingerprints:elasticNet",
        ),
        (
            ("raw[expression]+raw[mutations]", "fingerprints", "randomForest"),
            "raw[expression]+raw[mutations]:fingerprints:randomForest",
        ),
    ],
    ids=["predictor-only", "two-slot", "three-slot", "concat-slot"],
)
def test_model_recipe_is_formatted_from_slots(slots: tuple[str | None, str | None, str], expected: str) -> None:
    """Formatting writes the grammar that parsing reads.

    :param slots: ``(cell_line, drug, predictor)`` names to join.
    :param expected: Expected recipe string.
    """
    assert format_model_recipe(*slots) == expected


@pytest.mark.parametrize(
    "slots",
    [
        (None, None, "naiveMean"),
        ("scaledGeneExpression", None, "singleDrugElasticNet"),
        ("scaledGeneExpression", "fingerprints", "elasticNet"),
    ],
    ids=["predictor-only", "two-slot", "three-slot"],
)
def test_formatted_recipes_parse_back_to_the_same_slots(slots: tuple[str | None, str | None, str]) -> None:
    """A formatted recipe reads back as the slots it was written from.

    A concat slot has no round trip to check: it expands to a ``concatFeaturizers`` node whose
    name is not the recipe it came from, which is the same reason ``ModelConfig.model_id``
    cannot name one.

    :param slots: ``(cell_line, drug, predictor)`` names to join.
    """
    assert _slots(format_model_recipe(*slots)) == slots


def test_formatting_requires_a_predictor() -> None:
    with pytest.raises(ValueError, match="predictor is required"):
        format_model_recipe("scaledGeneExpression", "fingerprints", "")


def test_formatting_rejects_a_drug_slot_without_a_cell_line_slot() -> None:
    """A recipe fills its slots left to right, so this pair has no representation."""
    with pytest.raises(ValueError, match="cell_line is required when drug is set"):
        format_model_recipe(None, "fingerprints", "elasticNet")
