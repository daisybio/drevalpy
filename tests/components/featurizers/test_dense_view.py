"""Tests for the side-agnostic dense single-view featurizer base.

Mirrors :mod:`drevalpy.components.featurizers._dense_view`. Both entity sides are
exercised through the same base, because the two per-side copies this replaced had
one implementation between them. The branches nothing else in the suite reaches are
the ``fetch`` hit (a pre-computed variant registered in the MuData) and the
``_compute_from_source`` fallback taken when the declared view is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.drug.base import DenseViewDrugFeaturizer
from drevalpy.components.featurizers.sample.base import DenseViewSampleFeaturizer
from drevalpy.components.featurizers.storage import register_variant
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.feature_source import DrugFeatureSource, SampleFeatureSource
from tests.components.featurizers._name_alignment import assert_aligns_by_name, mock_source, random_views
from tests.conftest import MockFeatureSource
from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

_SAMPLE_PRECOMPUTED = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
_DRUG_PRECOMPUTED = np.array([[9.0, 8.0], [7.0, 6.0]], dtype=np.float32)


class _StoredDenseView(DenseViewSampleFeaturizer):
    """Dense view whose values were pre-computed into ``response.obsm``."""

    storage_key = "stored_dense"
    side = "sample"
    input_views = ("gene_expression",)


class _ComputedDenseView(DenseViewSampleFeaturizer):
    """Dense view with an on-the-fly fallback and no stored matrix."""

    precompute = True
    input_views = ("bionic_features",)

    def _compute_from_source(self, source, entity_ids: np.ndarray) -> np.ndarray:
        """Return a fixed-width matrix, standing in for a real computation."""
        return np.full((len(entity_ids), 4), 7.0, dtype=np.float32)


class _StrictDenseView(DenseViewSampleFeaturizer):
    """Dense view without a fallback, so a missing view must propagate."""

    input_views = ("bionic_features",)


class _StoredDrugView(DenseViewDrugFeaturizer):
    """Drug view whose values were pre-computed into ``response.varm``."""

    storage_key = "stored_view"
    side = "drug"
    input_views = ("morgan_fingerprint",)


class _ComputedDrugView(DenseViewDrugFeaturizer):
    """Drug view with an on-the-fly fallback and no stored matrix."""

    precompute = True
    input_views = ("morgan_fingerprint",)

    def _compute_from_source(self, source, entity_ids: np.ndarray) -> np.ndarray:
        """Return a fixed-width matrix, standing in for a real computation."""
        return np.full((len(entity_ids), 3), 5.0, dtype=np.float32)


class _NamedBlockDrugView(DenseViewDrugFeaturizer):
    """Drug view that declares its own output block name."""

    input_views = ("morgan_fingerprint",)
    output_block_specs = (BlockSpec("custom_block", None),)


def _sample_source() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2])},
            "cl2": {"gene_expression": np.array([0.3, 0.4])},
        },
        meta_info={"gene_expression": ["g1", "g2"]},
    )


def _drug_source() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "d1": {"morgan_fingerprint": np.array([1.0, 0.0])},
            "d2": {"morgan_fingerprint": np.array([0.0, 1.0])},
        },
        meta_info={"morgan_fingerprint": ["fp1", "fp2"]},
    )


@pytest.fixture
def stored_sample_source() -> SampleFeatureSource:
    """A dataset-backed source carrying a registered ``stored_dense`` variant."""
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    dataset.mdata.mod["response"].obsm["stored_dense_0"] = _SAMPLE_PRECOMPUTED
    register_variant(dataset.mdata, "stored_dense", "stored_dense_0", None, side="sample")
    return SampleFeatureSource(dataset, dataset.sample_ids)


@pytest.fixture
def stored_drug_source() -> DrugFeatureSource:
    """A dataset-backed source carrying a registered ``stored_view`` variant."""
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    dataset.mdata.mod["response"].varm["stored_view_0"] = _DRUG_PRECOMPUTED
    register_variant(dataset.mdata, "stored_view", "stored_view_0", None, side="drug")
    return DrugFeatureSource(dataset, dataset.drug_ids)


def test_dense_view_passes_the_declared_view_through() -> None:
    source = _sample_source()
    ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = _StrictDenseView(view="gene_expression").fit(source, entity_ids=ids)

    matrix = featurizer.transform(source, ids)

    assert featurizer.output_dim == 2
    np.testing.assert_allclose(matrix, [[0.1, 0.2], [0.3, 0.4]], rtol=1e-6)


def test_dense_view_passes_a_drug_view_through() -> None:
    source = _drug_source()
    ids = np.array(["d1", "d2"], dtype=str)
    featurizer = _StoredDrugView().fit(source, entity_ids=ids)

    matrix = featurizer.transform(source, ids)

    assert featurizer.output_dim == 2
    np.testing.assert_allclose(matrix, [[1.0, 0.0], [0.0, 1.0]])


def test_dense_view_defaults_to_the_single_declared_input_view() -> None:
    assert _StoredDenseView()._view == "gene_expression"
    assert _StoredDrugView()._view == "morgan_fingerprint"


def test_dense_view_names_its_block_after_the_view_and_carries_feature_names() -> None:
    source = _sample_source()
    ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = _StrictDenseView(view="gene_expression").fit(source, entity_ids=ids)

    blocks = featurizer.transform_blocks(source, ids)

    assert set(blocks) == {"gene_expression"}
    assert blocks["gene_expression"].feature_names == ("g1", "g2")


def test_dense_view_block_honours_declared_output_specs() -> None:
    source = _drug_source()
    ids = np.array(["d1", "d2"], dtype=str)
    featurizer = _NamedBlockDrugView().fit(source, entity_ids=ids)

    assert set(featurizer.transform_blocks(source, ids)) == {"custom_block"}


def test_dense_view_fit_uses_a_precomputed_variant_when_one_is_registered(
    stored_sample_source: SampleFeatureSource,
) -> None:
    featurizer = _StoredDenseView().fit(stored_sample_source, entity_ids=stored_sample_source.identifiers)

    assert featurizer.output_dim == _SAMPLE_PRECOMPUTED.shape[1]


def test_dense_view_transform_returns_the_precomputed_variant(
    stored_sample_source: SampleFeatureSource,
) -> None:
    featurizer = _StoredDenseView().fit(stored_sample_source, entity_ids=stored_sample_source.identifiers)

    matrix = featurizer.transform(stored_sample_source, stored_sample_source.identifiers)

    np.testing.assert_allclose(matrix, _SAMPLE_PRECOMPUTED)


def test_dense_view_reads_a_drug_side_variant_from_varm(stored_drug_source: DrugFeatureSource) -> None:
    featurizer = _StoredDrugView().fit(stored_drug_source, entity_ids=stored_drug_source.identifiers)

    matrix = featurizer.transform(stored_drug_source, stored_drug_source.identifiers)

    assert featurizer.output_dim == _DRUG_PRECOMPUTED.shape[1]
    np.testing.assert_allclose(matrix, _DRUG_PRECOMPUTED)


def test_dense_view_fit_falls_back_to_computing_from_source() -> None:
    featurizer = _ComputedDenseView().fit(_sample_source(), entity_ids=np.array(["cl1", "cl2"], dtype=str))

    assert featurizer.output_dim == 4


def test_dense_view_transform_falls_back_to_computing_from_source() -> None:
    source = _sample_source()
    ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = _ComputedDenseView().fit(source, entity_ids=ids)

    np.testing.assert_allclose(featurizer.transform(source, ids), np.full((2, 4), 7.0, dtype=np.float32))


def test_dense_view_drug_side_falls_back_to_computing_from_source() -> None:
    source = _drug_source()
    ids = np.array(["d1", "d2"], dtype=str)
    featurizer = _ComputedDrugView(view="chemberta").fit(source, entity_ids=ids)

    assert featurizer.output_dim == 3
    np.testing.assert_allclose(featurizer.transform(source, ids), np.full((2, 3), 5.0, dtype=np.float32))


def test_dense_view_without_a_fallback_propagates_the_missing_view() -> None:
    with pytest.raises(KeyError):
        _StrictDenseView().fit(_sample_source(), entity_ids=np.array(["cl1", "cl2"], dtype=str))


def test_dense_view_transform_without_a_fallback_propagates_the_missing_view() -> None:
    source = _sample_source()
    ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = _StrictDenseView(view="gene_expression").fit(source, entity_ids=ids)
    featurizer._view = "bionic_features"

    with pytest.raises(KeyError):
        featurizer.transform(source, ids)


def test_dense_view_fit_over_all_identifiers_when_none_are_given() -> None:
    assert _StrictDenseView(view="gene_expression").fit(_sample_source()).output_dim == 2
    assert _StoredDrugView().fit(_drug_source()).output_dim == 2


def test_dense_view_requires_fit_gate_is_off_by_default() -> None:
    source = _sample_source()
    ids = np.array(["cl1", "cl2"], dtype=str)

    matrix = _StrictDenseView(view="gene_expression")._transform_features(source, ids).matrix

    assert matrix.shape == (2, 2)


def test_dense_view_requires_fit_gate_rejects_an_unfitted_transform() -> None:
    class _NeedsFit(DenseViewSampleFeaturizer):
        input_views = ("gene_expression",)
        requires_fit = True

    with pytest.raises(RuntimeError, match="must be fit before transform"):
        _NeedsFit()._transform_features(_sample_source(), np.array(["cl1"], dtype=str))


def test_dense_view_fit_on_unique_ids_deduplicates_the_fit_rows() -> None:
    class _Unique(DenseViewSampleFeaturizer):
        input_views = ("gene_expression",)
        fit_on_unique_ids = True

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.seen: np.ndarray | None = None

        def _fit_state(self, source, entity_ids: np.ndarray) -> int:
            self.seen = entity_ids
            return super()._fit_state(source, entity_ids)

    featurizer = _Unique().fit(_sample_source(), entity_ids=np.array(["cl1", "cl1", "cl2"], dtype=str))

    assert list(featurizer.seen) == ["cl1", "cl2"]


class TestRestoreDenseState:
    """``_restore_dense_state`` is the shared tail of every subclass ``set_state``.

    Four sample featurizers used to repeat these three type-guarded reads
    verbatim around their own one fitted object, which is what made
    ``normalized_proteomics`` and ``scaled_gene_expression`` the repository's most
    co-changed duplication pair.
    """

    def test_it_restores_view_output_dim_and_the_fitted_flag(self) -> None:
        featurizer = _StrictDenseView(view="gene_expression")

        featurizer._restore_dense_state({"view": "mutations", "output_dim": 7, "fitted": True})

        assert featurizer._view == "mutations"
        assert featurizer._output_dim == 7
        assert featurizer._is_fitted is True

    def test_absent_keys_leave_the_fields_alone(self) -> None:
        """A subclass that never writes ``fitted`` must not have it reset for it."""
        featurizer = _StrictDenseView(view="gene_expression")
        featurizer._output_dim = 3
        featurizer._is_fitted = True

        featurizer._restore_dense_state({})

        assert featurizer._view == "gene_expression"
        assert featurizer._output_dim == 3
        assert featurizer._is_fitted is True

    def test_wrongly_typed_values_are_ignored(self) -> None:
        featurizer = _StrictDenseView(view="gene_expression")

        featurizer._restore_dense_state({"view": 5, "output_dim": "wide"})

        assert featurizer._view == "gene_expression"
        assert featurizer._output_dim == 0


def test_dense_view_reads_a_sample_view_by_name() -> None:
    assert_aligns_by_name(
        lambda: _StrictDenseView(view="gene_expression"), random_views({"gene_expression": list("abcdef")})
    )


def test_dense_view_blocks_carry_the_fitted_order_of_a_permuted_source() -> None:
    views = random_views({"gene_expression": ["a", "b", "c"]})
    source = mock_source(views)
    featurizer = _StrictDenseView(view="gene_expression").fit(source, entity_ids=source.identifiers)
    matrix, names = views["gene_expression"]
    permuted = mock_source({"gene_expression": (matrix[:, ::-1], names[::-1])})

    block = featurizer.transform_blocks(permuted, permuted.identifiers)["gene_expression"]

    assert block.feature_names == ("a", "b", "c")
    np.testing.assert_allclose(block.values, matrix, rtol=1e-6)


def test_dense_view_does_not_reindex_the_computed_fallback() -> None:
    source = _sample_source()
    featurizer = _ComputedDenseView(view="bionic_features").fit(source, entity_ids=np.array(["cl1", "cl2"]))

    assert featurizer._view_names is None
    assert featurizer.transform(source, np.array(["cl1"])).shape == (1, 4)


def test_dense_view_state_is_empty_before_fit_and_round_trips_after() -> None:
    views = random_views({"gene_expression": ["a", "b", "c"]})
    source = mock_source(views)
    featurizer = _StrictDenseView(view="gene_expression")
    assert featurizer.get_state() == {}

    featurizer.fit(source, entity_ids=source.identifiers)
    restored = _StrictDenseView(view="gene_expression")
    restored.set_state(featurizer.get_state())

    assert restored._view_names == ("a", "b", "c")
    assert restored.output_dim == 3
    assert restored._is_fitted
