"""Tests for landmark gene featurizers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from upath import UPath

from drevalpy.components.featurizers.sample import gene_lists
from drevalpy.components.featurizers.sample.gene_lists import resolve_gene_list
from drevalpy.components.featurizers.sample.landmark import (
    LandmarkGenesFeaturizer,
    LandmarkGenesReducedFeaturizer,
)
from drevalpy.errors import InvalidInputError
from tests.components.featurizers._name_alignment import assert_aligns_by_name, mock_source, random_views
from tests.components.featurizers.sample._helpers import assert_uses_precomputed_variant
from tests.conftest import MockFeatureSource

SYMBOLS = ("G1", "G2", "G3", "G4")


@pytest.fixture
def small_lists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    pd.DataFrame({"Symbol": SYMBOLS}).to_parquet(tmp_path / "landmark_genes.parquet", index=False)
    pd.DataFrame({"Symbol": SYMBOLS[:3]}).to_parquet(tmp_path / "landmark_genes_reduced.parquet", index=False)
    monkeypatch.setattr(gene_lists, "GENE_LISTS_DIR", UPath(tmp_path))
    return SYMBOLS


def _source(names: tuple[str, ...], values: dict[str, list[float]] | None = None) -> MockFeatureSource:
    values = values or {"cl1": list(range(1, len(names) + 1)), "cl2": list(range(len(names), 0, -1))}
    return MockFeatureSource(
        features={cl: {"gene_expression": np.array(row, dtype=np.float32)} for cl, row in values.items()},
        meta_info={"gene_expression": list(names)},
    )


def test_landmark_selects_the_list_in_list_order_and_persists_state(small_lists: tuple[str, ...]) -> None:
    features = _source((*reversed(small_lists), "EXTRA"), {"cl1": [1, 2, 3, 4, 5], "cl2": [5, 4, 3, 2, 1]})
    featurizer = LandmarkGenesFeaturizer(standardize=False, arcsinh=False)
    ids = np.array(["cl1", "cl2"])
    featurizer.fit(features, entity_ids=ids)
    assert featurizer.output_dim == 4
    matrix = featurizer.transform(features, ids)
    np.testing.assert_allclose(matrix, [[4, 3, 2, 1], [2, 3, 4, 5]])

    restored = LandmarkGenesFeaturizer()
    restored.set_state(featurizer.get_state())
    assert restored.output_dim == 4
    np.testing.assert_allclose(restored.transform(features, ids), matrix)


def test_landmark_reduced_uses_the_reduced_list(small_lists: tuple[str, ...]) -> None:
    featurizer = LandmarkGenesReducedFeaturizer(standardize=False)
    featurizer.fit(_source(small_lists), entity_ids=np.array(["cl1"]))
    assert featurizer.output_dim == 3


def test_landmark_reduced_works_on_the_shipped_list() -> None:
    symbols = resolve_gene_list("landmark_genes_reduced")
    source = mock_source(random_views({"gene_expression": symbols}, n_samples=3))
    featurizer = LandmarkGenesReducedFeaturizer().fit(source, entity_ids=source.identifiers)
    assert featurizer.output_dim == len(symbols)


def test_landmark_fit_errors(small_lists: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(InvalidInputError, match=r"view 'gene_expression' lacks 2 of 4 genes in 'landmark_genes'"):
        LandmarkGenesFeaturizer().fit(_source(small_lists[:2]), entity_ids=np.array(["cl1"]))
    unnamed = MockFeatureSource(features={"cl1": {"gene_expression": np.array([1.0, 2.0], dtype=np.float32)}})
    with pytest.raises(ValueError, match="no feature names for view"):
        LandmarkGenesFeaturizer().fit(unnamed, entity_ids=np.array(["cl1"]))
    pd.DataFrame({"other": ["A"]}).to_parquet(tmp_path / "landmark_genes.parquet", index=False)
    monkeypatch.setattr(gene_lists, "GENE_LISTS_DIR", UPath(tmp_path))
    with pytest.raises(ValueError, match="recognized gene-name column"):
        LandmarkGenesFeaturizer().fit(_source(("A",)), entity_ids=np.array(["cl1"]))


def test_landmark_scaling_options(small_lists: tuple[str, ...]) -> None:
    features = _source(small_lists, {"cl1": [1, 5, 2, 7], "cl2": [9, 2, 4, 1]})
    ids = np.array(["cl1", "cl2"])
    scaled = LandmarkGenesFeaturizer(standardize=True, minmax_scale=True).fit(features, entity_ids=ids)
    matrix = scaled.transform(features, ids)
    assert matrix.min() >= 0.0
    assert matrix.max() <= 1.0

    raw = _source(small_lists, {"cl1": [0, 1, 2, 3]})
    arcsinh = LandmarkGenesFeaturizer(standardize=False, arcsinh=True).fit(raw, entity_ids=ids[:1])
    np.testing.assert_allclose(arcsinh.transform(raw, ids[:1]), np.arcsinh([[0.0, 1.0, 2.0, 3.0]]), rtol=1e-6)


def test_landmark_serves_a_precomputed_variant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd.DataFrame({"Symbol": ["gene0", "gene1"]}).to_parquet(tmp_path / "landmark_genes.parquet", index=False)
    monkeypatch.setattr(gene_lists, "GENE_LISTS_DIR", UPath(tmp_path))
    assert_uses_precomputed_variant(LandmarkGenesFeaturizer())


def test_landmark_unfitted_featurizer() -> None:
    featurizer = LandmarkGenesFeaturizer()
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        featurizer._transform_features(_source(SYMBOLS), np.array(["cl1"]))
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        featurizer._empty_transform_features(_source(SYMBOLS))
    assert featurizer.get_state() == {}
    featurizer.set_state({"view_feature_names": ("A", "B", "C"), "output_dim": None, "fitted": True})
    assert featurizer.output_dim == 3


def test_landmark_blocks_carry_the_selected_gene_names_even_without_source_metadata(
    small_lists: tuple[str, ...],
) -> None:
    features = _source((*small_lists, "EXTRA"), {"cl1": [1, 2, 3, 4, 5]})
    ids = np.array(["cl1"])
    featurizer = LandmarkGenesFeaturizer(standardize=False).fit(features, entity_ids=ids)

    blocks = featurizer.transform_blocks(features, ids)

    assert set(blocks) == {"gene_expression"}
    assert blocks["gene_expression"].feature_names == small_lists
    unnamed = MockFeatureSource(features={"cl1": {"gene_expression": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}})
    assert featurizer.transform_blocks(unnamed, ids)["gene_expression"].feature_names == small_lists


def test_landmark_reads_genes_by_name_and_does_not_need_the_others(small_lists: tuple[str, ...]) -> None:
    fit_views = random_views({"gene_expression": [*small_lists, "NOT_A_GENE"]})
    assert_aligns_by_name(LandmarkGenesFeaturizer, fit_views)
    fit_source = mock_source(fit_views)
    featurizer = LandmarkGenesFeaturizer(standardize=False).fit(fit_source, entity_ids=fit_source.identifiers)
    without_extra = mock_source({"gene_expression": (fit_views["gene_expression"][0][:, :4], small_lists)})

    np.testing.assert_allclose(
        featurizer.transform(without_extra, without_extra.identifiers),
        featurizer.transform(fit_source, fit_source.identifiers),
    )


def _pair_fit(fit_on_pairs: bool, small_lists: tuple[str, ...], pairs: bool = True) -> LandmarkGenesFeaturizer:
    features = _source(small_lists, {"cl1": [1, 2, 3, 4], "cl2": [5, 8, 9, 20]})
    pair_ids = np.array(["cl1", "cl1", "cl1", "cl2"]) if pairs else None
    return LandmarkGenesFeaturizer(arcsinh=False, fit_on_pairs=fit_on_pairs).fit(
        features, entity_ids=np.array(["cl1", "cl2"]), pair_expanded_ids=pair_ids
    )


def test_landmark_fit_on_pairs_weights_by_pair_count_and_falls_back_without_pairs(small_lists: tuple[str, ...]) -> None:
    per_sample = [3.0, 5.0, 6.0, 12.0]
    np.testing.assert_allclose(_pair_fit(True, small_lists)._scaler.mean_, [2.0, 3.5, 4.5, 8.0])
    np.testing.assert_allclose(_pair_fit(False, small_lists)._scaler.mean_, per_sample)
    np.testing.assert_allclose(_pair_fit(True, small_lists, pairs=False)._scaler.mean_, per_sample)
    restored = LandmarkGenesFeaturizer()
    restored.set_state(_pair_fit(True, small_lists).get_state())
    assert restored._fit_on_pairs is True
