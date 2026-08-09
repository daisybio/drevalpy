"""Tests for SparseGO ontology metadata blocks."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.cell_line._sparsego_metadata import attach_sparsego_ontology_metadata
from drevalpy.components.featurizers.cell_line.sparsego_ontology import SparseGOOntologyFeaturizer
from tests.conftest import MockFeatureSource


def test_sparsego_ontology_emits_active_block_and_round_trips_state() -> None:
    features = MockFeatureSource(
        {"cl1": {"gene_expression": np.array([1.0, 2.0])}},
        meta_info={"gene_expression": ["a", "b"]},
    )
    attach_sparsego_ontology_metadata(
        features,
        {
            "layer_connections": [np.array([["term", "a"]])],
            "gene2id_mapping_ont": {"a": 0, "b": 1},
            "ontology_gene_order": ("a", "b"),
            "gene_dim_input": 2,
        },
    )
    featurizer = SparseGOOntologyFeaturizer().fit(features)
    block = featurizer.transform_blocks(features, np.array(["cl1"]))["gene_expression"]
    assert block.metadata is not None
    assert block.metadata["gene_dim_input"] == 2
    restored = SparseGOOntologyFeaturizer()
    restored.set_state(featurizer.get_state())
    np.testing.assert_allclose(restored.transform(features, np.array(["cl1"])), block.values)
