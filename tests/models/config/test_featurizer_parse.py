"""Tests for compact featurizer config parsing."""

from __future__ import annotations

import pytest

from drevalpy.models.config._featurizer_parse import normalize_featurizer_config


def test_normalize_string_shorthand() -> None:
    payload = normalize_featurizer_config("fingerprints", default_registry="drug")
    assert payload == {"name": "fingerprints", "registry": "drug"}


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


def test_normalize_plus_recipe_string() -> None:
    payload = normalize_featurizer_config(
        "scaledGeneExpression+raw[mutations]",
        default_registry="cell_line",
    )
    assert payload["name"] == "concatFeaturizers"
    assert payload["registry"] == "cell_line"
    children = payload["featurizers"]
    assert children[0]["name"] == "scaledGeneExpression"
    assert children[1]["name"] == "raw"
    assert children[1]["view"] == "mutations"
    assert all(child["registry"] == "cell_line" for child in children)


def test_normalize_plus_recipe_string_for_drug_registry() -> None:
    payload = normalize_featurizer_config("fingerprints+identity", default_registry="drug")
    children = payload["featurizers"]
    assert [child["name"] for child in children] == ["fingerprints", "identity"]
    assert all(child["registry"] == "drug" for child in children)


def test_normalize_rejects_empty_plus_recipe_piece() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalize_featurizer_config("scaledGeneExpression+", default_registry="cell_line")
    with pytest.raises(ValueError, match="non-empty"):
        normalize_featurizer_config("scaledGeneExpression++raw[mutations]", default_registry="cell_line")


def test_normalize_rejects_invalid_shape() -> None:
    with pytest.raises(TypeError, match="string, list, or mapping"):
        normalize_featurizer_config(123)


def test_normalize_bracket_atom_raw() -> None:
    payload = normalize_featurizer_config("raw[expression]", default_registry="cell_line")
    assert payload == {
        "name": "raw",
        "view": "gene_expression",
        "registry": "cell_line",
    }


def test_normalize_bracket_atom_pca() -> None:
    payload = normalize_featurizer_config("pca[proteomics]", default_registry="cell_line")
    assert payload["name"] == "pca"
    assert payload["view"] == "proteomics"


def test_normalize_bracket_plus_recipe() -> None:
    payload = normalize_featurizer_config(
        "raw[expression]+pca[proteomics]",
        default_registry="cell_line",
    )
    children = payload["featurizers"]
    assert children[0]["name"] == "raw"
    assert children[0]["view"] == "gene_expression"
    assert children[1]["name"] == "pca"
    assert children[1]["view"] == "proteomics"


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
        normalize_featurizer_config("raw", default_registry="cell_line")
    with pytest.raises(ValueError, match="requires an explicit view"):
        normalize_featurizer_config("pca", default_registry="cell_line")


def test_normalize_rejects_brackets_on_drug_registry() -> None:
    with pytest.raises(ValueError, match="cell-line featurizers"):
        normalize_featurizer_config("raw[expression]", default_registry="drug")


def test_normalize_rejects_unknown_view() -> None:
    with pytest.raises(ValueError, match="Unknown omics view"):
        normalize_featurizer_config("raw[not_a_view]", default_registry="cell_line")


def test_plus_inside_brackets_does_not_split_the_recipe() -> None:
    """A ``+`` within a view must stay in its atom, so the error names the bad view."""
    with pytest.raises(ValueError, match="Unknown omics view 'a\\+b'"):
        normalize_featurizer_config("raw[a+b]", default_registry="cell_line")


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


def test_legacy_hyperparameters_mapping_points_at_the_replacement() -> None:
    """The removed notation reports the one-key form that replaces it."""
    with pytest.raises(ValueError, match="no longer accept a nested 'hyperparameters' mapping"):
        normalize_featurizer_config(
            {"name": "pca", "view": "methylation", "hyperparameters": {"n_components": 8}},
            default_registry="cell_line",
        )


def test_legacy_hyperparameters_without_a_name_still_reports_the_replacement() -> None:
    """``hyperparameters`` is reserved, so it cannot be mistaken for a one-key shorthand."""
    with pytest.raises(ValueError, match="no longer accept a nested 'hyperparameters' mapping"):
        normalize_featurizer_config({"hyperparameters": {"n_components": 8}}, default_registry="cell_line")


def test_bracket_syntax_is_rejected_for_non_parametric_featurizers() -> None:
    with pytest.raises(ValueError, match="only supported for raw and pca"):
        normalize_featurizer_config("scaledGeneExpression[gene_expression]", default_registry="cell_line")


def test_non_atom_one_key_falls_back_to_the_registry_error() -> None:
    """A key that is not a single atom is looked up verbatim, so the registry reports it."""
    with pytest.raises(ValueError, match="Unknown Cell line featurizer: 'a\\+b'"):
        normalize_featurizer_config({"a+b": {"foo": 1}}, default_registry="cell_line")


def test_unparsable_one_key_falls_back_to_the_registry_error() -> None:
    """A key the grammar rejects outright is also looked up verbatim, not reported as syntax."""
    with pytest.raises(ValueError, match="Unknown Cell line featurizer: 'raw\\['"):
        normalize_featurizer_config({"raw[": {"foo": 1}}, default_registry="cell_line")


def test_one_key_loose_values_resolve_against_the_drug_registry() -> None:
    """No drug featurizer declares a space, so a loose value becomes a fixed option."""
    payload = normalize_featurizer_config({"fingerprints": {"n_bits": 512}}, default_registry="drug")
    assert payload["registry"] == "drug"
    assert payload["options"] == {"n_bits": 512}
    assert "hyperparameter_space" not in payload


def test_named_mapping_without_a_view_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires an explicit view"):
        normalize_featurizer_config({"name": "pca"}, default_registry="cell_line")


def test_named_mapping_with_a_blank_view_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires an explicit view"):
        normalize_featurizer_config({"name": "pca", "view": "  "}, default_registry="cell_line")


def test_empty_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        normalize_featurizer_config("   ", default_registry="cell_line")


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
    with pytest.raises(ValueError, match="string, one-key mapping, or dict with 'name'"):
        normalize_featurizer_config({"view": "methylation", "options": {}}, default_registry="cell_line")
