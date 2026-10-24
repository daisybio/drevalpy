"""Tests for the SparseGO ontology metadata read helper (``_sparsego_metadata``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from drevalpy.components.featurizers.sample._sparsego_metadata import read_sparsego_ontology_metadata
from drevalpy.types.data.feature_source import SampleFeatureSource
from tests.conftest import MockFeatureSource
from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints


def _metadata() -> dict[str, Any]:
    return {
        "layer_connections": [np.array([["term", "a"]])],
        "gene2id_mapping_ont": {"a": 0, "b": 1},
        "ontology_gene_order": ("a", "b"),
        "gene_dim_input": 2,
    }


def test_read_resolves_metadata_from_the_source() -> None:
    source = MockFeatureSource(
        {"cl1": {"gene_expression": np.array([1.0, 2.0])}}, meta_info={"sparsego_ontology": _metadata()}
    )

    restored = read_sparsego_ontology_metadata(source)

    assert restored is not None
    assert restored["gene_dim_input"] == 2
    assert restored["gene2id_mapping_ont"] == {"a": 0, "b": 1}
    assert restored["ontology_gene_order"] == ("a", "b")
    np.testing.assert_array_equal(restored["layer_connections"][0], np.array([["term", "a"]]))


def test_read_returns_none_without_usable_metadata() -> None:
    assert read_sparsego_ontology_metadata(MockFeatureSource({"cl1": {"gene_expression": np.array([1.0])}})) is None
    assert (
        read_sparsego_ontology_metadata(MockFeatureSource({}, meta_info={"sparsego_ontology": {"gene_dim_input": 2}}))
        is None
    )
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    assert read_sparsego_ontology_metadata(SampleFeatureSource(dataset, dataset.sample_ids)) is None


def test_read_resolves_metadata_stored_in_mdata_uns() -> None:
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    dataset.mdata.uns["sparsego_ontology"] = _metadata()

    restored = read_sparsego_ontology_metadata(SampleFeatureSource(dataset, dataset.sample_ids))

    assert restored is not None
    assert restored["gene_dim_input"] == 2
