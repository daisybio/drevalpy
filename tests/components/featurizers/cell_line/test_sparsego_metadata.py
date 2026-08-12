"""Tests for SparseGO ontology metadata attach/read helpers.

Mirrors :mod:`drevalpy.components.featurizers.cell_line._sparsego_metadata`. The
featurizer that consumes this metadata is covered in ``test_sparsego_ontology.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from drevalpy.components.featurizers.cell_line._sparsego_metadata import (
    attach_sparsego_ontology_metadata,
    read_sparsego_ontology_metadata,
)
from tests.conftest import MockFeatureSource


def _metadata() -> dict[str, Any]:
    return {
        "layer_connections": [np.array([["term", "a"]])],
        "gene2id_mapping_ont": {"a": 0, "b": 1},
        "ontology_gene_order": ("a", "b"),
        "gene_dim_input": 2,
    }


def test_attach_then_read_round_trips_metadata() -> None:
    source = MockFeatureSource({"cl1": {"gene_expression": np.array([1.0, 2.0])}})

    attach_sparsego_ontology_metadata(source, _metadata())
    restored = read_sparsego_ontology_metadata(source)

    assert restored is not None
    assert restored["gene_dim_input"] == 2
    assert restored["gene2id_mapping_ont"] == {"a": 0, "b": 1}
    assert restored["ontology_gene_order"] == ("a", "b")
    np.testing.assert_array_equal(restored["layer_connections"][0], np.array([["term", "a"]]))


def test_read_returns_none_without_attached_metadata() -> None:
    source = MockFeatureSource({"cl1": {"gene_expression": np.array([1.0, 2.0])}})

    assert read_sparsego_ontology_metadata(source) is None


def test_read_tolerates_a_source_that_raises_on_unknown_metadata_keys() -> None:
    from drevalpy.types.data.feature_source import CellLineFeatureSource
    from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

    dataset = synthetic_mudataset_gene_expression_fingerprints()
    source = CellLineFeatureSource(dataset, dataset.cell_line_ids)

    assert read_sparsego_ontology_metadata(source) is None


def test_read_ignores_payload_without_layer_connections() -> None:
    source = MockFeatureSource({}, meta_info={"sparsego_ontology": {"gene_dim_input": 2}})

    assert read_sparsego_ontology_metadata(source) is None


def test_attach_writes_to_mdata_uns_for_dataset_backed_sources() -> None:
    from drevalpy.types.data.feature_source import CellLineFeatureSource
    from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

    dataset = synthetic_mudataset_gene_expression_fingerprints()
    source = CellLineFeatureSource(dataset, dataset.cell_line_ids)

    attach_sparsego_ontology_metadata(source, _metadata())

    assert dataset.mdata.uns["sparsego_ontology"]["gene_dim_input"] == 2
    assert read_sparsego_ontology_metadata(source) is not None


def test_attach_rejects_sources_without_a_metadata_home() -> None:
    class _Opaque:
        pass

    with pytest.raises(TypeError, match="Cannot attach metadata"):
        attach_sparsego_ontology_metadata(_Opaque(), _metadata())
