"""Tests for the dense single-view cell-line featurizer base.

Mirrors :mod:`drevalpy.components.featurizers.cell_line.dense_view`. The two
branches nothing else in the suite reaches are the ``fetch`` hit (a pre-computed
variant registered in the MuData) and the ``_compute_from_source`` fallback taken
when the declared view is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.cell_line.dense_view import DenseViewCellLineFeaturizer
from drevalpy.components.featurizers.storage import register_variant
from drevalpy.types.data.feature_source import CellLineFeatureSource
from tests.conftest import MockFeatureSource
from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

_PRECOMPUTED = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)


class _StoredDenseView(DenseViewCellLineFeaturizer):
    """Dense view whose values were pre-computed into ``response.obsm``."""

    storage_key = "stored_dense"
    side = "cell_line"
    input_views = ("gene_expression",)


class _ComputedDenseView(DenseViewCellLineFeaturizer):
    """Dense view with an on-the-fly fallback and no stored matrix."""

    precompute = True
    input_views = ("bionic_features",)

    def _compute_from_source(self, source, entity_ids: np.ndarray) -> np.ndarray:
        """Return a fixed-width matrix, standing in for a real computation."""
        return np.full((len(entity_ids), 4), 7.0, dtype=np.float32)


class _StrictDenseView(DenseViewCellLineFeaturizer):
    """Dense view without a fallback, so a missing view must propagate."""

    input_views = ("bionic_features",)


def _dense_source() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2])},
            "cl2": {"gene_expression": np.array([0.3, 0.4])},
        },
        meta_info={"gene_expression": ["g1", "g2"]},
    )


@pytest.fixture
def stored_source() -> CellLineFeatureSource:
    """A dataset-backed source carrying a registered ``stored_dense`` variant."""
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    mdata = dataset.mdata
    mdata.mod["response"].obsm["stored_dense_0"] = _PRECOMPUTED
    register_variant(mdata, "stored_dense", "stored_dense_0", None, side="cell_line")
    return CellLineFeatureSource(dataset, dataset.cell_line_ids)


def test_dense_view_passes_the_declared_view_through() -> None:
    source = _dense_source()
    ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = _StrictDenseView(view="gene_expression").fit(source, entity_ids=ids)

    matrix = featurizer.transform(source, ids)

    assert featurizer.output_dim == 2
    np.testing.assert_allclose(matrix, [[0.1, 0.2], [0.3, 0.4]], rtol=1e-6)


def test_dense_view_defaults_to_the_single_declared_input_view() -> None:
    featurizer = _StoredDenseView()

    assert featurizer._view == "gene_expression"


def test_dense_view_names_its_block_after_the_view_and_carries_feature_names() -> None:
    source = _dense_source()
    ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = _StrictDenseView(view="gene_expression").fit(source, entity_ids=ids)

    blocks = featurizer.transform_blocks(source, ids)

    assert set(blocks) == {"gene_expression"}
    assert blocks["gene_expression"].feature_names == ("g1", "g2")


def test_dense_view_fit_uses_a_precomputed_variant_when_one_is_registered(
    stored_source: CellLineFeatureSource,
) -> None:
    featurizer = _StoredDenseView().fit(stored_source, entity_ids=stored_source.identifiers)

    assert featurizer.output_dim == _PRECOMPUTED.shape[1]


def test_dense_view_transform_returns_the_precomputed_variant(stored_source: CellLineFeatureSource) -> None:
    featurizer = _StoredDenseView().fit(stored_source, entity_ids=stored_source.identifiers)

    matrix = featurizer.transform(stored_source, stored_source.identifiers)

    np.testing.assert_allclose(matrix, _PRECOMPUTED)


def test_dense_view_fit_falls_back_to_computing_from_source() -> None:
    source = _dense_source()

    featurizer = _ComputedDenseView().fit(source, entity_ids=np.array(["cl1", "cl2"], dtype=str))

    assert featurizer.output_dim == 4


def test_dense_view_transform_falls_back_to_computing_from_source() -> None:
    source = _dense_source()
    ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = _ComputedDenseView().fit(source, entity_ids=ids)

    matrix = featurizer.transform(source, ids)

    np.testing.assert_allclose(matrix, np.full((2, 4), 7.0, dtype=np.float32))


def test_dense_view_without_a_fallback_propagates_the_missing_view() -> None:
    source = _dense_source()
    ids = np.array(["cl1", "cl2"], dtype=str)

    with pytest.raises(KeyError):
        _StrictDenseView().fit(source, entity_ids=ids)


def test_dense_view_transform_without_a_fallback_propagates_the_missing_view() -> None:
    source = _dense_source()
    ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = _StrictDenseView(view="gene_expression")
    featurizer.fit(source, entity_ids=ids)
    featurizer._view = "bionic_features"

    with pytest.raises(KeyError):
        featurizer.transform(source, ids)


def test_dense_view_fit_over_all_identifiers_when_none_are_given() -> None:
    source = _dense_source()

    featurizer = _StrictDenseView(view="gene_expression").fit(source)

    assert featurizer.output_dim == 2
