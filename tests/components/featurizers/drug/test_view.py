"""Tests for the single-view drug featurizer.

Mirrors :mod:`drevalpy.components.featurizers.drug.view`. The two branches nothing
else in the suite reaches are the ``fetch`` hit (a pre-computed variant registered
in ``response.varm``) and the ``_compute_from_source`` fallback taken when the
declared view is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.featurizers.storage import register_variant
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.feature_source import DrugFeatureSource
from tests.conftest import MockFeatureSource
from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

_PRECOMPUTED = np.array([[9.0, 8.0], [7.0, 6.0]], dtype=np.float32)


class _StoredView(ViewDrugFeaturizer):
    """Drug view whose values were pre-computed into ``response.varm``."""

    storage_key = "stored_view"
    side = "drug"


class _ComputedView(ViewDrugFeaturizer):
    """Drug view with an on-the-fly fallback and no stored matrix."""

    precompute = True

    def _compute_from_source(self, source, entity_ids: np.ndarray) -> np.ndarray:
        """Return a fixed-width matrix, standing in for a real computation."""
        return np.full((len(entity_ids), 3), 5.0, dtype=np.float32)


class _NamedBlockView(ViewDrugFeaturizer):
    """Drug view that declares its own output block name."""

    output_block_specs = (BlockSpec("custom_block", None),)


def _drug_source() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "d1": {"morgan_fingerprint": np.array([1.0, 0.0])},
            "d2": {"morgan_fingerprint": np.array([0.0, 1.0])},
        },
        meta_info={"morgan_fingerprint": ["fp1", "fp2"]},
    )


@pytest.fixture
def stored_source() -> DrugFeatureSource:
    """A dataset-backed source carrying a registered ``stored_view`` variant."""
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    mdata = dataset.mdata
    mdata.mod["response"].varm["stored_view_0"] = _PRECOMPUTED
    register_variant(mdata, "stored_view", "stored_view_0", None, side="drug")
    return DrugFeatureSource(dataset, dataset.drug_ids)


def test_view_drug_featurizer_passes_the_declared_view_through() -> None:
    source = _drug_source()
    ids = np.array(["d1", "d2"], dtype=str)
    featurizer = ViewDrugFeaturizer().fit(source, entity_ids=ids)

    matrix = featurizer.transform(source, ids)

    assert featurizer.output_dim == 2
    np.testing.assert_allclose(matrix, [[1.0, 0.0], [0.0, 1.0]])


def test_view_drug_featurizer_block_defaults_to_the_view_name() -> None:
    source = _drug_source()
    ids = np.array(["d1", "d2"], dtype=str)
    featurizer = ViewDrugFeaturizer().fit(source, entity_ids=ids)

    blocks = featurizer.transform_blocks(source, ids)

    assert set(blocks) == {"morgan_fingerprint"}
    assert blocks["morgan_fingerprint"].feature_names == ("fp1", "fp2")


def test_view_drug_featurizer_block_honours_declared_output_specs() -> None:
    source = _drug_source()
    ids = np.array(["d1", "d2"], dtype=str)
    featurizer = _NamedBlockView().fit(source, entity_ids=ids)

    blocks = featurizer.transform_blocks(source, ids)

    assert set(blocks) == {"custom_block"}


def test_view_drug_featurizer_fit_uses_a_precomputed_variant(stored_source: DrugFeatureSource) -> None:
    featurizer = _StoredView().fit(stored_source, entity_ids=stored_source.identifiers)

    assert featurizer.output_dim == _PRECOMPUTED.shape[1]


def test_view_drug_featurizer_transform_returns_the_precomputed_variant(
    stored_source: DrugFeatureSource,
) -> None:
    featurizer = _StoredView().fit(stored_source, entity_ids=stored_source.identifiers)

    matrix = featurizer.transform(stored_source, stored_source.identifiers)

    np.testing.assert_allclose(matrix, _PRECOMPUTED)


def test_view_drug_featurizer_fit_falls_back_to_computing_from_source() -> None:
    featurizer = _ComputedView(view="chemberta").fit(_drug_source(), entity_ids=np.array(["d1", "d2"], dtype=str))

    assert featurizer.output_dim == 3


def test_view_drug_featurizer_transform_falls_back_to_computing_from_source() -> None:
    source = _drug_source()
    ids = np.array(["d1", "d2"], dtype=str)
    featurizer = _ComputedView(view="chemberta").fit(source, entity_ids=ids)

    matrix = featurizer.transform(source, ids)

    np.testing.assert_allclose(matrix, np.full((2, 3), 5.0, dtype=np.float32))


def test_view_drug_featurizer_without_a_fallback_propagates_the_missing_view() -> None:
    with pytest.raises(KeyError):
        ViewDrugFeaturizer(view="chemberta").fit(_drug_source(), entity_ids=np.array(["d1"], dtype=str))


def test_view_drug_featurizer_transform_without_a_fallback_propagates_the_missing_view() -> None:
    source = _drug_source()
    ids = np.array(["d1", "d2"], dtype=str)
    featurizer = ViewDrugFeaturizer().fit(source, entity_ids=ids)
    featurizer._view = "chemberta"

    with pytest.raises(KeyError):
        featurizer.transform(source, ids)


def test_view_drug_featurizer_fit_over_all_identifiers_when_none_are_given() -> None:
    featurizer = ViewDrugFeaturizer().fit(_drug_source())

    assert featurizer.output_dim == 2
