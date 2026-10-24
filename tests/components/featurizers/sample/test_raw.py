"""Tests for raw sample featurizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drevalpy.components.featurizers.sample.raw import RawSampleFeaturizer
from drevalpy.errors import InvalidInputError
from tests.components.featurizers.sample._helpers import assert_uses_precomputed_variant, use_gene_lists
from tests.conftest import MockFeatureSource


def _make_features() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {
                "gene_expression": np.array([1.0, 2.0], dtype=np.float32),
                "mutations": np.array([0.0, 1.0], dtype=np.float32),
            },
            "cl2": {
                "gene_expression": np.array([3.0, 4.0], dtype=np.float32),
                "mutations": np.array([1.0, 0.0], dtype=np.float32),
            },
        },
        meta_info={"gene_expression": ["a", "b"], "mutations": ["a", "b"]},
    )


def test_raw_passes_through_view_and_requires_one() -> None:
    features = _make_features()
    featurizer = RawSampleFeaturizer(view="gene_expression")
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    np.testing.assert_allclose(featurizer.transform(features, entity_ids), [[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="requires an explicit view"):
        RawSampleFeaturizer(view="")


def test_raw_prefers_a_precomputed_variant_over_the_view() -> None:
    assert_uses_precomputed_variant(RawSampleFeaturizer(view="gene_expression"))


def test_raw_gene_list_keeps_the_listed_genes_in_list_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["b"]})
    features = _make_features()
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = RawSampleFeaturizer(view="mutations", gene_list="mine").fit(features, entity_ids=entity_ids)

    blocks = featurizer.transform_blocks(features, entity_ids)

    assert blocks["mutations"].feature_names == ("b",)
    np.testing.assert_allclose(blocks["mutations"].values, [[1.0], [0.0]])
    assert RawSampleFeaturizer(view="mutations")._fetch_hyperparameters() is None
    assert featurizer._fetch_hyperparameters() == {"gene_list": "mine"}


def test_raw_gene_list_raises_when_genes_are_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["a", "z"]})
    with pytest.raises(InvalidInputError, match="lacks 1 of 2 genes in 'mine'"):
        RawSampleFeaturizer(view="mutations", gene_list="mine").fit(
            _make_features(), entity_ids=np.array(["cl1"], dtype=str)
        )
