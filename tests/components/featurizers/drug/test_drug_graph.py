"""Tests for graph drug featurizer payload handling and on-the-fly fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

import drevalpy.components.featurizers.drug.drug_graph as drug_graph_module
from drevalpy.components.featurizers.drug.drug_graph import (
    DrugGraphFeaturizer,
    _one_hot_encode,
    _smiles_to_graph,
)
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import DrugFeatureSource
from tests._import_shims import block_imports
from tests.conftest import MockFeatureSource

_D1 = np.array(["d1"])


def _graph(nodes: int = 2) -> Data:
    return Data(x=torch.ones((nodes, 3)), edge_index=torch.tensor([[0], [1]], dtype=torch.long))


class _NoStoredGraphSource(DrugFeatureSource):
    """``get_entity_view`` raises for an absent view; returning ``None`` is the only way to reach the fallback."""

    def get_entity_view(self, entity_id: str, view: str) -> None:
        return None


def test_drug_graph_preserves_payloads_and_infers_node_feature_width() -> None:
    graph, other = _graph(), Data(x=torch.zeros((1, 3)), edge_index=torch.empty((2, 0), dtype=torch.long))
    features = MockFeatureSource({"d1": {"drug_graph": graph}, "d2": {"drug_graph": other}})
    featurizer = DrugGraphFeaturizer().fit(features, entity_ids=_D1)

    block = featurizer.transform_blocks(features, _D1)["drug_graph"]

    assert block.values.shape == (1,)
    assert block.values[0] is graph
    assert featurizer.output_dim == 3
    assert set(featurizer._graphs) == {"d1"}
    assert featurizer.transform(features, np.array(["d2"]))[0] is other
    assert set(DrugGraphFeaturizer.get_hyperparameter_space()) == {"add_hydrogens"}


def test_drug_graph_skips_unresolvable_drugs_at_fit_and_raises_at_transform() -> None:
    features = MockFeatureSource({"d1": {}})
    featurizer = DrugGraphFeaturizer().fit(features, entity_ids=_D1)

    assert featurizer._graphs == {}
    assert featurizer.output_dim == 0
    with pytest.raises(KeyError, match="graph computation failed"):
        featurizer.transform(features, _D1)
    assert DrugGraphFeaturizer()._compute_from_source(
        MockFeatureSource({"d1": {}, "d2": {}}), np.array(["d1", "d2"])
    ).tolist() == [None, None]


def test_drug_graph_computes_graphs_on_the_fly_for_missing_views(synthetic_dataset: Dataset) -> None:
    source = _NoStoredGraphSource(synthetic_dataset, synthetic_dataset.drug_ids)
    drug_ids = synthetic_dataset.drug_ids[:2]

    featurizer = DrugGraphFeaturizer().fit(source, entity_ids=drug_ids)

    assert set(featurizer._graphs) == set(drug_ids)
    assert featurizer.output_dim > 0
    assert isinstance(DrugGraphFeaturizer().transform(source, drug_ids[:1])[0], Data)


def test_drug_graph_computes_graphs_from_dataset_smiles(synthetic_dataset: Dataset) -> None:
    source = DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)

    payloads = DrugGraphFeaturizer()._compute_from_source(source, synthetic_dataset.drug_ids[:2])

    assert payloads.shape == (2,)
    assert all(isinstance(payload, Data) for payload in payloads)


def test_drug_graph_returns_none_for_a_non_string_smiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drug_graph_module, "get_smiles_for_entities", lambda s, ids: pd.Series({"d1": float("nan")}))

    assert DrugGraphFeaturizer()._compute_graph_from_smiles_for_entity(MockFeatureSource({"d1": {}}), "d1") is None


def test_smiles_to_graph_handles_unparseable_single_atom_and_hydrogens() -> None:
    assert _smiles_to_graph("this is not a molecule") is None
    graph = _smiles_to_graph("C")
    assert graph is not None
    assert graph.x.shape[0] == 1
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, 6)
    with_hs = _smiles_to_graph("C", add_hydrogens=True)
    assert with_hs is not None
    assert with_hs.x.shape[0] == graph.x.shape[0] + 4
    assert with_hs.edge_index.shape[1] == 8


def test_one_hot_encode_uses_the_trailing_bin_for_unknown_values() -> None:
    assert _one_hot_encode("z", ["a", "b"]) == [0, 0, 1]
    assert _one_hot_encode("b", ["a", "b"]) == [0, 1, 0]


@pytest.mark.parametrize(
    ("blocked", "message"),
    [("rdkit", "rdkit is required"), ("torch", "torch and torch_geometric are required")],
)
def test_smiles_to_graph_names_the_missing_dependency(
    monkeypatch: pytest.MonkeyPatch, blocked: str, message: str
) -> None:
    block_imports(monkeypatch, blocked)

    with pytest.raises(ImportError, match=message):
        _smiles_to_graph("CCO")
