"""Tests for PharmaFormer gene preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drevalpy.components.featurizers.sample.pharmaformer_gene_expression import (
    PharmaFormerGeneExpressionFeaturizer,
)
from drevalpy.errors import InvalidInputError
from tests.components.featurizers._name_alignment import assert_aligns_by_name, random_views
from tests.components.featurizers.sample._helpers import assert_uses_precomputed_variant, use_gene_lists
from tests.conftest import MockFeatureSource


def _features() -> MockFeatureSource:
    return MockFeatureSource(
        {f"cl{i}": {"gene_expression": np.array([i, i + 2], dtype=np.float32)} for i in range(3)},
        meta_info={"gene_expression": ["a", "b"]},
    )


def test_pharmaformer_gene_expression_round_trips_state() -> None:
    features = _features()
    featurizer = PharmaFormerGeneExpressionFeaturizer()
    assert featurizer.output_dim == 0
    assert featurizer.get_state() == {}
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        featurizer._transform_features(features, np.array(["cl0"]))
    featurizer.fit(features, pair_expanded_ids=np.array(["cl0", "cl1", "cl0"]), pair_expanded_es_ids=np.array(["cl2"]))
    blocks = featurizer.transform_blocks(features, np.array(["cl0", "cl2"]))
    assert blocks["gene_expression"].feature_names == ("a", "b")
    restored = PharmaFormerGeneExpressionFeaturizer()
    restored.set_state(featurizer.get_state())
    np.testing.assert_allclose(restored.transform(features, np.array(["cl0", "cl2"])), blocks["gene_expression"].values)


def test_pharmaformer_gene_expression_fits_the_scaler_on_unique_samples_or_every_sample() -> None:
    featurizer = PharmaFormerGeneExpressionFeaturizer().fit(
        _features(),
        entity_ids=np.array(["cl0", "cl1"]),
        pair_expanded_ids=np.array(["cl0", "cl0", "cl0", "cl1"]),
    )
    np.testing.assert_allclose(featurizer._scaler.mean_, [0.5, 2.5])
    np.testing.assert_allclose(PharmaFormerGeneExpressionFeaturizer().fit(_features())._scaler.mean_, [1.0, 3.0])


def test_pharmaformer_gene_expression_serves_a_precomputed_variant() -> None:
    assert_uses_precomputed_variant(PharmaFormerGeneExpressionFeaturizer(), ids_kwarg="pair_expanded_ids")


def test_pharmaformer_gene_expression_reads_columns_by_name() -> None:
    assert_aligns_by_name(
        PharmaFormerGeneExpressionFeaturizer,
        random_views({"gene_expression": list("abcdef")}),
        fit_kwargs=lambda ids: {"pair_expanded_ids": ids},
    )


def test_pharmaformer_gene_expression_gene_list_selects_and_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["b"]})
    features = _features()
    ids = np.array(["cl0", "cl1", "cl0"])
    featurizer = PharmaFormerGeneExpressionFeaturizer(gene_list="mine").fit(features, pair_expanded_ids=ids)

    blocks = featurizer.transform_blocks(features, np.array(["cl0", "cl2"]))

    assert featurizer.output_dim == 1
    assert blocks["gene_expression"].feature_names == ("b",)
    restored = PharmaFormerGeneExpressionFeaturizer()
    restored.set_state(featurizer.get_state())
    assert restored._gene_list == "mine"
    assert PharmaFormerGeneExpressionFeaturizer()._fetch_hyperparameters() is None
    assert featurizer._fetch_hyperparameters() == {"gene_list": "mine"}
    np.testing.assert_allclose(restored.transform(features, np.array(["cl0", "cl2"])), blocks["gene_expression"].values)


def test_pharmaformer_gene_expression_gene_list_raises_when_genes_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["a", "z"]})
    with pytest.raises(InvalidInputError, match="lacks 1 of 2 genes in 'mine'"):
        PharmaFormerGeneExpressionFeaturizer(gene_list="mine").fit(_features(), pair_expanded_ids=np.array(["cl0"]))
