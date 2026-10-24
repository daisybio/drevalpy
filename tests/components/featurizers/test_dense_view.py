"""Tests for the side-agnostic dense single-view featurizer base (``_dense_view``)."""

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
    storage_key = "stored_dense"
    side = "sample"
    input_views = ("gene_expression",)


class _ComputedDenseView(DenseViewSampleFeaturizer):
    precompute = True
    input_views = ("bionic_features",)

    def _compute_from_source(self, source, entity_ids: np.ndarray) -> np.ndarray:
        return np.full((len(entity_ids), 4), 7.0, dtype=np.float32)


class _StrictDenseView(DenseViewSampleFeaturizer):
    input_views = ("bionic_features",)


class _StoredDrugView(DenseViewDrugFeaturizer):
    storage_key = "stored_view"
    side = "drug"
    input_views = ("morgan_fingerprint",)


class _ComputedDrugView(DenseViewDrugFeaturizer):
    precompute = True
    input_views = ("morgan_fingerprint",)

    def _compute_from_source(self, source, entity_ids: np.ndarray) -> np.ndarray:
        return np.full((len(entity_ids), 3), 5.0, dtype=np.float32)


class _NamedBlockDrugView(DenseViewDrugFeaturizer):
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


_IDS = np.array(["cl1", "cl2"], dtype=str)


@pytest.fixture
def stored_sample_source() -> SampleFeatureSource:
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    dataset.mdata.mod["response"].obsm["stored_dense_0"] = _SAMPLE_PRECOMPUTED
    register_variant(dataset.mdata, "stored_dense", "stored_dense_0", None, side="sample")
    return SampleFeatureSource(dataset, dataset.sample_ids)


@pytest.fixture
def stored_drug_source() -> DrugFeatureSource:
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    dataset.mdata.mod["response"].varm["stored_view_0"] = _DRUG_PRECOMPUTED
    register_variant(dataset.mdata, "stored_view", "stored_view_0", None, side="drug")
    return DrugFeatureSource(dataset, dataset.drug_ids)


def test_dense_view_passes_the_declared_view_through_with_names() -> None:
    source = _sample_source()
    featurizer = _StrictDenseView(view="gene_expression").fit(source, entity_ids=_IDS)

    assert featurizer.output_dim == 2
    np.testing.assert_allclose(featurizer.transform(source, _IDS), [[0.1, 0.2], [0.3, 0.4]], rtol=1e-6)
    blocks = featurizer.transform_blocks(source, _IDS)
    assert set(blocks) == {"gene_expression"}
    assert blocks["gene_expression"].feature_names == ("g1", "g2")


def test_dense_view_drug_side_passes_through_and_honours_declared_block_specs() -> None:
    source = _drug_source()
    ids = np.array(["d1", "d2"], dtype=str)
    featurizer = _StoredDrugView().fit(source, entity_ids=ids)

    assert featurizer._view == "morgan_fingerprint"
    np.testing.assert_allclose(featurizer.transform(source, ids), [[1.0, 0.0], [0.0, 1.0]])
    assert set(_NamedBlockDrugView().fit(source, entity_ids=ids).transform_blocks(source, ids)) == {"custom_block"}


@pytest.mark.parametrize(
    ("featurizer_cls", "fixture", "expected"),
    [
        (_StoredDenseView, "stored_sample_source", _SAMPLE_PRECOMPUTED),
        (_StoredDrugView, "stored_drug_source", _DRUG_PRECOMPUTED),
    ],
)
def test_dense_view_uses_a_registered_precomputed_variant(request, featurizer_cls, fixture, expected) -> None:
    source = request.getfixturevalue(fixture)
    featurizer = featurizer_cls().fit(source, entity_ids=source.identifiers)

    assert featurizer.output_dim == expected.shape[1]
    np.testing.assert_allclose(featurizer.transform(source, source.identifiers), expected)


@pytest.mark.parametrize(
    ("featurizer", "source", "ids", "width", "fill"),
    [
        (_ComputedDenseView(), _sample_source(), _IDS, 4, 7.0),
        (_ComputedDrugView(view="chemberta"), _drug_source(), np.array(["d1", "d2"]), 3, 5.0),
    ],
)
def test_dense_view_falls_back_to_computing_from_source(featurizer, source, ids, width, fill) -> None:
    featurizer.fit(source, entity_ids=ids)

    assert featurizer.output_dim == width
    np.testing.assert_allclose(featurizer.transform(source, ids), np.full((2, width), fill, dtype=np.float32))
    assert featurizer._view_names is None


def test_dense_view_without_a_fallback_propagates_the_missing_view() -> None:
    source = _sample_source()
    with pytest.raises(KeyError):
        _StrictDenseView().fit(source, entity_ids=_IDS)
    featurizer = _StrictDenseView(view="gene_expression").fit(source, entity_ids=_IDS)
    featurizer._view = "bionic_features"
    with pytest.raises(KeyError):
        featurizer.transform(source, _IDS)


def test_dense_view_fit_over_all_identifiers_when_none_are_given() -> None:
    assert _StrictDenseView(view="gene_expression").fit(_sample_source()).output_dim == 2
    assert _StoredDrugView().fit(_drug_source()).output_dim == 2


def test_dense_view_requires_fit_gate() -> None:
    class _NeedsFit(DenseViewSampleFeaturizer):
        input_views = ("gene_expression",)
        requires_fit = True

    featurizer = _StrictDenseView(view="gene_expression")
    assert featurizer._transform_features(_sample_source(), _IDS).matrix.shape == (2, 2)
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


def test_restore_dense_state_restores_fields_and_ignores_absent_or_mistyped_keys() -> None:
    featurizer = _StrictDenseView(view="gene_expression")
    featurizer._restore_dense_state({"view": 5, "output_dim": "wide"})
    assert (featurizer._view, featurizer._output_dim) == ("gene_expression", 0)

    featurizer._restore_dense_state({"view": "mutations", "output_dim": 7, "fitted": True})
    assert (featurizer._view, featurizer._output_dim, featurizer._is_fitted) == ("mutations", 7, True)

    featurizer._restore_dense_state({})
    assert (featurizer._view, featurizer._output_dim, featurizer._is_fitted) == ("mutations", 7, True)


def test_dense_view_reads_a_sample_view_by_name() -> None:
    assert_aligns_by_name(
        lambda: _StrictDenseView(view="gene_expression"), random_views({"gene_expression": list("abcdef")})
    )


def test_dense_view_blocks_carry_the_fitted_order_and_state_round_trips() -> None:
    views = random_views({"gene_expression": ["a", "b", "c"]})
    source = mock_source(views)
    featurizer = _StrictDenseView(view="gene_expression")
    assert featurizer.get_state() == {}
    featurizer.fit(source, entity_ids=source.identifiers)
    matrix, names = views["gene_expression"]
    permuted = mock_source({"gene_expression": (matrix[:, ::-1], names[::-1])})

    block = featurizer.transform_blocks(permuted, permuted.identifiers)["gene_expression"]

    assert block.feature_names == ("a", "b", "c")
    np.testing.assert_allclose(block.values, matrix, rtol=1e-6)
    restored = _StrictDenseView(view="gene_expression")
    restored.set_state(featurizer.get_state())
    assert (restored._view_names, restored.output_dim, restored._is_fitted) == (("a", "b", "c"), 3, True)
