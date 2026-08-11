"""Tests for featurizer mapping normalization.

Recipe strings are expanded by ``drevalpy.models.config._recipe`` before they reach the
normalizer, so recipe notation itself is covered in ``test_recipe.py``. What is tested here is
the mapping side, plus the fact that an expanded recipe and the equivalent YAML normalize
identically.
"""

from __future__ import annotations

import pytest

from drevalpy.models.config._featurizer_parse import normalize_featurizer_config
from drevalpy.models.config._recipe import expand_featurizer_recipe


def _normalize_recipe(recipe: str, *, default_registry: str) -> dict:
    """Normalize a recipe string the way a config slot does.

    :param recipe: Recipe string to expand and normalize.
    :param default_registry: Registry to resolve names against.
    :returns: Normalized featurizer mapping.
    """
    return normalize_featurizer_config(
        expand_featurizer_recipe(recipe),
        default_registry=default_registry,
    )


def test_normalize_named_mapping() -> None:
    payload = normalize_featurizer_config({"name": "fingerprints"}, default_registry="drug")
    assert payload == {"name": "fingerprints", "registry": "drug"}


def test_normalize_rejects_a_recipe_string() -> None:
    """The normalizer takes mappings; a recipe is expanded before it gets here."""
    with pytest.raises(TypeError, match="list or mapping"):
        normalize_featurizer_config("fingerprints", default_registry="drug")


def test_normalize_list_shorthand() -> None:
    payload = normalize_featurizer_config(
        ["scaledGeneExpression", "raw[mutations]"],
        default_registry="cell_line",
    )
    assert payload["name"] == "concatFeaturizers"
    assert payload["registry"] == "cell_line"
    children = payload["featurizers"]
    assert children[0]["name"] == "scaledGeneExpression"
    assert children[1]["name"] == "raw"
    assert children[1]["view"] == "mutations"
    assert all(child["registry"] == "cell_line" for child in children)


def test_normalize_list_with_parameterized_child() -> None:
    payload = normalize_featurizer_config(
        [
            "scaledGeneExpression",
            {"pca[methylation]": {"n_components": 64}},
        ],
        default_registry="cell_line",
    )
    children = payload["featurizers"]
    assert children[1]["name"] == "pca"
    assert children[1]["view"] == "methylation"
    assert children[1]["hyperparameter_space"]["n_components"]["default"] == 64


def test_normalize_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalize_featurizer_config([], default_registry="cell_line")


def test_a_recipe_and_the_equivalent_yaml_normalize_alike() -> None:
    """The two notations are documented as interchangeable, down to the resolved view."""
    from_recipe = _normalize_recipe("raw[expression]+pca[methylation]", default_registry="cell_line")
    from_yaml = normalize_featurizer_config(
        {
            "name": "concatFeaturizers",
            "featurizers": [
                {"name": "raw", "view": "expression"},
                {"name": "pca", "view": "methylation"},
            ],
        },
        default_registry="cell_line",
    )
    assert from_recipe == from_yaml
    assert [child["view"] for child in from_recipe["featurizers"]] == ["expression", "methylation"]


def test_normalize_rejects_invalid_shape() -> None:
    with pytest.raises(TypeError, match="list or mapping"):
        normalize_featurizer_config(123)


def test_normalize_records_the_registry_on_every_child() -> None:
    payload = _normalize_recipe("fingerprints+identity", default_registry="drug")
    children = payload["featurizers"]
    assert [child["name"] for child in children] == ["fingerprints", "identity"]
    assert all(child["registry"] == "drug" for child in children)


def test_normalize_resolves_a_view_alias_written_out_in_full() -> None:
    """A spelled-out view is passed through unchanged (alias resolution removed)."""
    payload = normalize_featurizer_config({"name": "raw", "view": "expression"}, default_registry="cell_line")
    assert payload == {
        "name": "raw",
        "view": "expression",
        "registry": "cell_line",
    }


def test_normalize_leaves_an_already_canonical_view_alone() -> None:
    payload = normalize_featurizer_config({"name": "pca", "view": "proteomics"}, default_registry="cell_line")
    assert payload["view"] == "proteomics"


def test_normalize_leaves_a_custom_view_untouched() -> None:
    """A view may name a matrix shipped with a dataset, which is no alias and no typo."""
    payload = normalize_featurizer_config({"name": "raw", "view": "custom_test_view"}, default_registry="cell_line")
    assert payload["view"] == "custom_test_view"


def test_a_bracket_and_an_explicit_view_field_are_indistinguishable() -> None:
    """A bracket is shorthand for the field, so neither is treated more strictly than the other."""
    for view in ("expression", "custom_test_view"):
        assert _normalize_recipe(f"raw[{view}]", default_registry="cell_line") == normalize_featurizer_config(
            {"name": "raw", "view": view},
            default_registry="cell_line",
        )


def test_normalize_leaves_a_view_alone_for_featurizers_that_are_not_view_parametric() -> None:
    """Elsewhere ``view`` names an output block, which is not an omics alias."""
    payload = normalize_featurizer_config({"name": "fingerprints", "view": "ignored"}, default_registry="drug")
    assert payload["view"] == "ignored"


def test_normalize_one_key_mapping_with_brackets() -> None:
    payload = normalize_featurizer_config(
        {"pca[methylation]": {"n_components": 64}},
        default_registry="cell_line",
    )
    assert payload["name"] == "pca"
    assert payload["view"] == "methylation"
    assert payload["hyperparameter_space"]["n_components"]["default"] == 64


def test_normalize_rejects_bare_raw_or_pca() -> None:
    with pytest.raises(ValueError, match="requires an explicit view"):
        _normalize_recipe("raw", default_registry="cell_line")
    with pytest.raises(ValueError, match="requires an explicit view"):
        _normalize_recipe("pca", default_registry="cell_line")


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
def test_one_key_body_folds_loose_values(
    body: dict,
    expected_space_default: int | None,
    expected_options: dict | None,
) -> None:
    """A loose value moves a declared default; anything undeclared becomes a fixed option.

    :param body: Body of the one-key mapping form.
    :param expected_space_default: Expected ``n_components`` default, when relevant.
    :param expected_options: Expected ``options`` mapping, when relevant.
    """
    payload = normalize_featurizer_config({"pca[methylation]": body}, default_registry="cell_line")
    if expected_space_default is not None:
        assert payload["hyperparameter_space"]["n_components"]["default"] == expected_space_default
    if expected_options is not None:
        assert payload["options"] == expected_options


def test_non_atom_one_key_falls_back_to_the_registry_error() -> None:
    """A key that is not a single atom is looked up verbatim, so the registry reports it."""
    with pytest.raises(ValueError, match="Unknown Cell line featurizer: 'a\\+b'"):
        normalize_featurizer_config({"a+b": {"foo": 1}}, default_registry="cell_line")


def test_unparsable_one_key_falls_back_to_the_registry_error() -> None:
    """A key the grammar rejects outright is also looked up verbatim, not reported as syntax."""
    with pytest.raises(ValueError, match="Unknown Cell line featurizer: 'raw\\['"):
        normalize_featurizer_config({"raw[": {"foo": 1}}, default_registry="cell_line")


def test_one_key_loose_values_resolve_against_the_drug_registry() -> None:
    """Fingerprints declares n_bits in its HP space, so the value overrides the default."""
    payload = normalize_featurizer_config({"fingerprints": {"n_bits": 512}}, default_registry="drug")
    assert payload["registry"] == "drug"
    assert payload["hyperparameter_space"]["n_bits"]["default"] == 512


def test_named_mapping_without_a_view_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires an explicit view"):
        normalize_featurizer_config({"name": "pca"}, default_registry="cell_line")


def test_named_mapping_with_a_blank_view_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires an explicit view"):
        normalize_featurizer_config({"name": "pca", "view": "  "}, default_registry="cell_line")


def test_one_key_with_a_bracket_records_the_view_without_judging_it() -> None:
    """A bracketed key is read structurally, so a view lands on whatever featurizer was named."""
    payload = normalize_featurizer_config(
        {"scaledGeneExpression[gene_expression]": {"foo": 1}},
        default_registry="cell_line",
    )
    assert payload["name"] == "scaledGeneExpression"
    assert payload["view"] == "gene_expression"
    assert payload["options"] == {"foo": 1}


def test_one_key_body_must_be_a_mapping() -> None:
    with pytest.raises(ValueError, match="must be a mapping when provided"):
        normalize_featurizer_config({"scaledGeneExpression": 5}, default_registry="cell_line")


def test_one_key_body_may_be_null() -> None:
    payload = normalize_featurizer_config({"scaledGeneExpression": None}, default_registry="cell_line")
    assert payload == {"name": "scaledGeneExpression", "registry": "cell_line"}


def test_one_key_body_may_declare_children() -> None:
    payload = normalize_featurizer_config(
        {"concatFeaturizers": {"featurizers": ["scaledGeneExpression", "raw[mutations]"]}},
        default_registry="cell_line",
    )
    assert payload["name"] == "concatFeaturizers"
    assert [child["name"] for child in payload["featurizers"]] == ["scaledGeneExpression", "raw"]


def test_children_must_be_a_list() -> None:
    with pytest.raises(ValueError, match="featurizers must be a list when set"):
        normalize_featurizer_config(
            {"name": "concatFeaturizers", "featurizers": "scaledGeneExpression"},
            default_registry="cell_line",
        )


def test_mapping_without_name_or_one_key_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="list, one-key mapping, or dict with 'name'"):
        normalize_featurizer_config({"view": "methylation", "options": {}}, default_registry="cell_line")
