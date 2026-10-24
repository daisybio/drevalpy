"""Tests for featurizer mapping normalization (recipe strings are expanded first, see ``test_recipe.py``)."""

from __future__ import annotations

import pytest

from drevalpy.models.config._featurizer_parse import normalize_featurizer_config
from drevalpy.models.config._recipe import expand_featurizer_recipe


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
