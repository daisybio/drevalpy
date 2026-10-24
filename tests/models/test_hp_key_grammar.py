"""Tests for the qualified hyperparameter key grammar shared by ``models.config`` and ``models.tuning``."""

from __future__ import annotations

import pytest

from drevalpy.models._hp_key_grammar import (
    DRUG_SLOT,
    FEATURIZER_SLOTS,
    PREDICTOR_SLOT,
    REGISTRY_TO_SLOT,
    SAMPLE_SLOT,
    SLOT_TO_REGISTRY,
    featurizer_prefix,
    is_featurizer_slot_key,
    predictor_prefix,
    reject_indexed_featurizer_key,
    split_predictor_key,
    split_prefixed_key,
)


def test_slot_constants_are_the_wire_format() -> None:
    assert (SAMPLE_SLOT, DRUG_SLOT, PREDICTOR_SLOT) == ("sample_featurizer", "drug_featurizer", "predictor")
    assert FEATURIZER_SLOTS == (SAMPLE_SLOT, DRUG_SLOT)
    assert REGISTRY_TO_SLOT == {"sample": SAMPLE_SLOT, "drug": DRUG_SLOT}
    assert SLOT_TO_REGISTRY == {slot: registry for registry, slot in REGISTRY_TO_SLOT.items()}


@pytest.mark.parametrize(
    ("registry", "selector", "expected"),
    [
        ("sample", "pca", "sample_featurizer.pca.n_components"),
        ("sample", "pca[methylation]", "sample_featurizer.pca[methylation].n_components"),
        ("drug", "fingerprints", "drug_featurizer.fingerprints.n_components"),
    ],
)
def test_prefix_builders(registry: str, selector: str, expected: str) -> None:
    assert featurizer_prefix(registry, selector, "n_components") == expected
    assert predictor_prefix("elasticNet", "alpha") == "predictor.elasticNet.alpha"
    with pytest.raises(KeyError):
        featurizer_prefix("proteome", "pca", "n_components")


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("sample_featurizer.pca.n_components", True),
        ("drug_featurizer.fingerprints.radius", True),
        ("predictor.elasticNet.alpha", False),
        ("n_components", False),
        ("sample_featurizer", False),
    ],
)
def test_is_featurizer_slot_key(key: str, expected: bool) -> None:
    assert is_featurizer_slot_key(key) is expected


@pytest.mark.parametrize(
    "key",
    [
        "sample_featurizer.pca.0.n_components",
        "drug_featurizer.fingerprints.1.radius",
        "sample_featurizer.pca.12.n_components",
        "sample_featurizer.pca.0.nested.param",
    ],
)
def test_indexed_featurizer_keys_are_rejected(key: str) -> None:
    with pytest.raises(ValueError, match="no longer supported"):
        reject_indexed_featurizer_key(key)
    with pytest.raises(ValueError, match=r"sample_featurizer\.pca\[<view>\]\.n_components"):
        reject_indexed_featurizer_key("sample_featurizer.pca.0.n_components")


@pytest.mark.parametrize(
    "key",
    [
        "sample_featurizer.pca.n_components",
        "sample_featurizer.pca[gene_expression].n_components",
        "sample_featurizer.pca.view.n_components",
        "predictor.elasticNet.0.alpha",
        "sample_featurizer.pca.0",
    ],
)
def test_other_keys_are_not_rejected_as_indexed(key: str) -> None:
    assert reject_indexed_featurizer_key(key) is None


@pytest.mark.parametrize(
    ("registry", "selector", "param"),
    [
        ("sample", "pca", "n_components"),
        ("sample", "pca[methylation]", "n_components"),
        ("drug", "fingerprints", "radius"),
        ("drug", "fingerprints", "nested.param"),
    ],
)
def test_split_prefixed_key_round_trips_a_built_key(registry: str, selector: str, param: str) -> None:
    assert split_prefixed_key(featurizer_prefix(registry, selector, param)) == (registry, selector, param)


@pytest.mark.parametrize("key", ["predictor.elasticNet.alpha", "n_components", "sample_featurizer.pca"])
def test_split_prefixed_key_returns_none_for_a_non_featurizer_key(key: str) -> None:
    assert split_prefixed_key(key) is None


def test_split_prefixed_key_propagates_the_indexed_key_rejection() -> None:
    with pytest.raises(ValueError, match="no longer supported"):
        split_prefixed_key("sample_featurizer.pca.0.n_components")


@pytest.mark.parametrize(("name", "param"), [("elasticNet", "alpha"), ("randomForest", "nested.param")])
def test_split_predictor_key_round_trips_a_built_key(name: str, param: str) -> None:
    assert split_predictor_key(predictor_prefix(name, param)) == (name, param)


@pytest.mark.parametrize("key", ["sample_featurizer.pca.n_components", "predictor.elasticNet", "alpha"])
def test_split_predictor_key_returns_none_for_a_non_predictor_key(key: str) -> None:
    assert split_predictor_key(key) is None
