"""Tests for compact featurizer config parsing."""

from __future__ import annotations

import pytest

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config


def test_normalize_string_shorthand() -> None:
    payload = normalize_featurizer_config("fingerprints", default_registry="drug")
    assert payload == {"name": "fingerprints", "hyperparameters": {}, "registry": "drug"}


def test_normalize_list_shorthand() -> None:
    payload = normalize_featurizer_config(
        ["scaledGeneExpression", "raw[mutations]"],
        default_registry="cell_line",
    )
    assert payload["name"] == "concatFeaturizers"
    assert payload["registry"] == "cell_line"
    children = payload["hyperparameters"]["featurizers"]
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
    children = payload["hyperparameters"]["featurizers"]
    assert children[1]["name"] == "pca"
    assert children[1]["view"] == "methylation"
    assert children[1]["hyperparameters"]["n_components"] == 64


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
    children = payload["hyperparameters"]["featurizers"]
    assert children[0]["name"] == "scaledGeneExpression"
    assert children[1]["name"] == "raw"
    assert children[1]["view"] == "mutations"
    assert all(child["registry"] == "cell_line" for child in children)


def test_normalize_plus_recipe_string_for_drug_registry() -> None:
    payload = normalize_featurizer_config("fingerprints+identity", default_registry="drug")
    children = payload["hyperparameters"]["featurizers"]
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
        "hyperparameters": {},
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
    children = payload["hyperparameters"]["featurizers"]
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
    assert payload["hyperparameters"]["n_components"] == 64


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
