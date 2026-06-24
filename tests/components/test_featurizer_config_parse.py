"""Tests for compact featurizer config parsing."""

from __future__ import annotations

import pytest

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config


def test_normalize_string_shorthand() -> None:
    payload = normalize_featurizer_config("fingerprints", default_registry="drug")
    assert payload == {"name": "fingerprints", "hyperparameters": {}, "registry": "drug"}


def test_normalize_one_key_mapping() -> None:
    payload = normalize_featurizer_config(
        {"concatFeaturizers": {"featurizers": ["scaledGeneExpression", "mutations"]}},
        default_registry="cell_line",
    )
    assert payload["name"] == "concatFeaturizers"
    assert payload["hyperparameters"]["featurizers"][0]["name"] == "scaledGeneExpression"


def test_normalize_nested_parameterized_child() -> None:
    payload = normalize_featurizer_config(
        {
            "concatFeaturizers": {
                "featurizers": [
                    "scaledGeneExpression",
                    {"methylationPCA": {"n_components": 64}},
                ],
            },
        },
        default_registry="cell_line",
    )
    children = payload["hyperparameters"]["featurizers"]
    assert children[1]["name"] == "methylationPCA"
    assert children[1]["hyperparameters"]["n_components"] == 64


def test_normalize_rejects_invalid_shape() -> None:
    with pytest.raises(TypeError, match="string or mapping"):
        normalize_featurizer_config(123)
