"""Tests for compact featurizer config parsing."""

from __future__ import annotations

import pytest

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config


def test_normalize_string_shorthand() -> None:
    payload = normalize_featurizer_config("fingerprints", default_registry="drug")
    assert payload == {"name": "fingerprints", "hyperparameters": {}, "registry": "drug"}


def test_normalize_list_shorthand() -> None:
    payload = normalize_featurizer_config(
        ["scaledGeneExpression", "mutations"],
        default_registry="cell_line",
    )
    assert payload["name"] == "concatFeaturizers"
    assert payload["registry"] == "cell_line"
    children = payload["hyperparameters"]["featurizers"]
    assert [child["name"] for child in children] == ["scaledGeneExpression", "mutations"]
    assert all(child["registry"] == "cell_line" for child in children)


def test_normalize_list_with_parameterized_child() -> None:
    payload = normalize_featurizer_config(
        [
            "scaledGeneExpression",
            {"methylationPCA": {"n_components": 64}},
        ],
        default_registry="cell_line",
    )
    children = payload["hyperparameters"]["featurizers"]
    assert children[1]["name"] == "methylationPCA"
    assert children[1]["hyperparameters"]["n_components"] == 64


def test_normalize_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalize_featurizer_config([], default_registry="cell_line")


def test_normalize_plus_recipe_string() -> None:
    payload = normalize_featurizer_config(
        "scaledGeneExpression+mutations",
        default_registry="cell_line",
    )
    assert payload["name"] == "concatFeaturizers"
    assert payload["registry"] == "cell_line"
    children = payload["hyperparameters"]["featurizers"]
    assert [child["name"] for child in children] == ["scaledGeneExpression", "mutations"]
    assert all(child["registry"] == "cell_line" for child in children)


def test_normalize_plus_recipe_string_for_drug_registry() -> None:
    payload = normalize_featurizer_config("fingerprints+oneHot", default_registry="drug")
    children = payload["hyperparameters"]["featurizers"]
    assert [child["name"] for child in children] == ["fingerprints", "oneHot"]
    assert all(child["registry"] == "drug" for child in children)


def test_normalize_rejects_empty_plus_recipe_piece() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalize_featurizer_config("scaledGeneExpression+", default_registry="cell_line")
    with pytest.raises(ValueError, match="non-empty"):
        normalize_featurizer_config("scaledGeneExpression++mutations", default_registry="cell_line")


def test_normalize_rejects_invalid_shape() -> None:
    with pytest.raises(TypeError, match="string, list, or mapping"):
        normalize_featurizer_config(123)
