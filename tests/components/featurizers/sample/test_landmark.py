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
    """Point the shipped lists at a directory holding four-gene stand-ins."""
    pd.DataFrame({"Symbol": SYMBOLS}).to_csv(tmp_path / "landmark_genes.csv", index=False)
    pd.DataFrame({"Symbol": SYMBOLS[:3]}).to_csv(tmp_path / "landmark_genes_reduced.csv", index=False)
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
    views = random_views({"gene_expression": symbols}, n_samples=3)
    source = mock_source(views)
    featurizer = LandmarkGenesReducedFeaturizer().fit(source, entity_ids=source.identifiers)
    assert featurizer.output_dim == len(symbols)


def test_landmark_raises_with_the_count_when_the_view_lacks_genes(small_lists: tuple[str, ...]) -> None:
    with pytest.raises(InvalidInputError, match=r"view 'gene_expression' lacks 2 of 4 genes in 'landmark_genes'"):
        LandmarkGenesFeaturizer().fit(_source(small_lists[:2]), entity_ids=np.array(["cl1"]))


def test_landmark_fails_clearly_on_bad_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd.DataFrame({"other": ["A"]}).to_csv(tmp_path / "landmark_genes.csv", index=False)
    monkeypatch.setattr(gene_lists, "GENE_LISTS_DIR", UPath(tmp_path))
    with pytest.raises(ValueError, match="recognized gene-name column"):
        LandmarkGenesFeaturizer().fit(_source(("A",)), entity_ids=np.array(["cl1"]))


def test_landmark_requires_feature_names_on_the_view() -> None:
    features = MockFeatureSource(
        features={"cl1": {"gene_expression": np.array([1.0, 2.0], dtype=np.float32)}},
    )

    with pytest.raises(ValueError, match="no feature names for view"):
        LandmarkGenesFeaturizer().fit(features, entity_ids=np.array(["cl1"]))


def test_landmark_minmax_scaling_bounds_output_to_the_unit_interval(small_lists: tuple[str, ...]) -> None:
    features = _source(small_lists, {"cl1": [1, 5, 2, 7], "cl2": [9, 2, 4, 1]})
    ids = np.array(["cl1", "cl2"])
    featurizer = LandmarkGenesFeaturizer(standardize=True, minmax_scale=True).fit(features, entity_ids=ids)

    matrix = featurizer.transform(features, ids)

    assert matrix.min() >= 0.0
    assert matrix.max() <= 1.0


def test_landmark_without_standardization_keeps_raw_arcsinh_values(small_lists: tuple[str, ...]) -> None:
    features = _source(small_lists, {"cl1": [0, 1, 2, 3]})
    ids = np.array(["cl1"])
    featurizer = LandmarkGenesFeaturizer(standardize=False, arcsinh=True).fit(features, entity_ids=ids)

    matrix = featurizer.transform(features, ids)

    np.testing.assert_allclose(matrix, np.arcsinh([[0.0, 1.0, 2.0, 3.0]]), rtol=1e-6)


def test_landmark_serves_a_precomputed_variant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd.DataFrame({"Symbol": ["gene0", "gene1"]}).to_csv(tmp_path / "landmark_genes.csv", index=False)
    monkeypatch.setattr(gene_lists, "GENE_LISTS_DIR", UPath(tmp_path))
    assert_uses_precomputed_variant(LandmarkGenesFeaturizer())


def test_landmark_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        LandmarkGenesFeaturizer()._transform_features(_source(SYMBOLS), np.array(["cl1"]))


def test_landmark_transform_blocks_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        LandmarkGenesFeaturizer()._empty_transform_features(_source(SYMBOLS))


def test_landmark_state_is_empty_before_fit() -> None:
    assert LandmarkGenesFeaturizer().get_state() == {}


def test_landmark_hyperparameter_space_exposes_the_two_scaling_flags() -> None:
    assert set(LandmarkGenesFeaturizer.get_hyperparameter_space()) == {"standardize", "minmax_scale"}


def test_landmark_blocks_carry_the_selected_gene_names(small_lists: tuple[str, ...]) -> None:
    features = _source((*small_lists, "EXTRA"), {"cl1": [1, 2, 3, 4, 5]})
    ids = np.array(["cl1"])
    featurizer = LandmarkGenesFeaturizer(standardize=False).fit(features, entity_ids=ids)

    blocks = featurizer.transform_blocks(features, ids)

    assert set(blocks) == {"gene_expression"}
    assert blocks["gene_expression"].feature_names == small_lists


def test_landmark_set_state_derives_output_dim_from_gene_names() -> None:
    featurizer = LandmarkGenesFeaturizer()

    featurizer.set_state({"view_feature_names": ("A", "B", "C"), "output_dim": None, "fitted": True})

    assert featurizer.output_dim == 3


def test_landmark_blocks_keep_the_fitted_gene_names_on_a_source_without_metadata(
    small_lists: tuple[str, ...],
) -> None:
    """Fit needs feature names, but a later source is free not to expose any."""
    named = _source(small_lists, {"cl1": [1, 2, 3, 4]})
    unnamed = MockFeatureSource(features={"cl1": {"gene_expression": np.array([1.0, 2.0, 3.0, 4.0])}})
    ids = np.array(["cl1"])
    featurizer = LandmarkGenesFeaturizer(standardize=False).fit(named, entity_ids=ids)

    blocks = featurizer.transform_blocks(unnamed, ids)

    assert blocks["gene_expression"].feature_names == small_lists


def test_landmark_reads_genes_by_name_and_ignores_the_others(small_lists: tuple[str, ...]) -> None:
    views = random_views({"gene_expression": [*small_lists, "NOT_A_GENE"]})

    assert_aligns_by_name(LandmarkGenesFeaturizer, views)


def test_landmark_does_not_need_genes_outside_its_list(small_lists: tuple[str, ...]) -> None:
    fit_views = random_views({"gene_expression": [*small_lists, "NOT_A_GENE"]})
    fit_source = mock_source(fit_views)
    featurizer = LandmarkGenesFeaturizer(standardize=False).fit(fit_source, entity_ids=fit_source.identifiers)
    without_extra = mock_source({"gene_expression": (fit_views["gene_expression"][0][:, :4], small_lists)})

    np.testing.assert_allclose(
        featurizer.transform(without_extra, without_extra.identifiers),
        featurizer.transform(fit_source, fit_source.identifiers),
    )
