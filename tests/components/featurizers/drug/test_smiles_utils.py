"""Tests for SMILES lookup through a ``FeatureSource`` (``_smiles_utils``)."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.drug._smiles_utils import as_pyg_graph, get_smiles_for_entities
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import DrugFeatureSource
from tests.conftest import MockFeatureSource
from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints


def test_returns_none_without_a_mudata_backing_or_smiles_column() -> None:
    assert get_smiles_for_entities(MockFeatureSource(features={"d1": {}}), np.array(["d1"])) is None
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    source = DrugFeatureSource(dataset, dataset.drug_ids)
    assert get_smiles_for_entities(source, dataset.drug_ids) is None


def test_returns_smiles_indexed_by_the_requested_drug_ids(synthetic_dataset: Dataset) -> None:
    drug_ids = synthetic_dataset.drug_ids[:3]
    source = DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)

    smiles = get_smiles_for_entities(source, drug_ids)

    assert smiles is not None
    assert list(smiles.index) == list(drug_ids)
    assert all(isinstance(value, str) and value for value in smiles)
    unknown = get_smiles_for_entities(source, np.array(["not-a-drug"]))
    assert unknown is not None and unknown.isna().all()


def test_portable_graph_conversion_preserves_arrays_and_index_dtype() -> None:
    import torch

    payload = {"x": np.array([[1.0, 2.0], [3.0, 4.0]]), "edge_index": np.array([[0, 1], [1, 0]])}
    graph = as_pyg_graph(payload)
    assert graph.edge_index.dtype == torch.long
    np.testing.assert_array_equal(graph.x.numpy(), payload["x"])
    assert isinstance(payload["x"], np.ndarray)
    assert as_pyg_graph(graph) is graph
    assert as_pyg_graph(None) is None
