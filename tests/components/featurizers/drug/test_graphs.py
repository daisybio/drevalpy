"""Tests for the graph-based drug featurizers: DrugGraph payloads and the MolGNet graph conversion and network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem
from torch_geometric.data import Data

import drevalpy.components.featurizers.drug.drug_graph as drug_graph_module
from drevalpy.components.featurizers.drug._molgnet_network import (
    AddSegId,
    BertLayerNorm,
    LinearActivation,
    MolGNet,
    SelfLoop,
    _gelu,
    _MessagePassing,
    atom_cumsum,
    bond_cumsum,
    mol_to_graph_data_obj_complex,
)
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


def _mol_graph(smiles: str = "CCO"):
    return mol_to_graph_data_obj_complex(Chem.MolFromSmiles(smiles))


def _tiny_model() -> MolGNet:
    torch.manual_seed(0)
    model = MolGNet(num_layer=1, emb_dim=8, heads=2, num_message_passing=1, drop_ratio=0.0)
    return model.eval()


def test_cumulative_offsets_are_monotonic() -> None:
    assert np.all(np.diff(atom_cumsum) > 0)
    assert np.all(np.diff(bond_cumsum) > 0)


@pytest.mark.parametrize(
    ("smiles", "nodes", "edges"),
    [("CCO", 3, 4), ("C", 1, 0), ("c1ccccc1", 6, 12)],
    ids=["ethanol", "single-atom", "benzene"],
)
def test_mol_to_graph_encodes_atoms_and_both_bond_directions(smiles: str, nodes: int, edges: int) -> None:
    graph = _mol_graph(smiles)

    assert graph.x.shape == (nodes, 8)
    assert graph.x.dtype == torch.long
    assert graph.edge_index.shape == (2, edges)
    assert graph.edge_attr.shape == (edges, 5)


def test_mol_to_graph_rejects_none() -> None:
    with pytest.raises(ValueError, match="must not be None"):
        mol_to_graph_data_obj_complex(None)


def test_self_loop_and_seg_id_transforms() -> None:
    graph = _mol_graph("CCO")
    before_edges = graph.edge_index.shape[1]

    looped = SelfLoop()(graph)
    segmented = AddSegId()(looped)

    assert looped.edge_index.shape[1] == looped.edge_attr.shape[0] == before_edges + 3
    assert segmented.node_seg.tolist() == [0, 0, 0]
    assert segmented.edge_seg.shape[0] == segmented.num_edges
    assert torch.all(segmented.edge_seg == 0)


def test_add_seg_id_rejects_a_graph_without_a_node_count() -> None:
    class _NodelessGraph:
        num_nodes = None
        num_edges = 0

    with pytest.raises(ValueError, match="graph reports no node count"):
        AddSegId()(_NodelessGraph())


def test_molgnet_forward_returns_one_embedding_row_per_atom_from_a_graph_or_tensors() -> None:
    graph = AddSegId()(SelfLoop()(_mol_graph("CCO")))
    model = _tiny_model()

    with torch.no_grad():
        embedding = model(graph)
        unpacked = model(graph.x, graph.edge_index, graph.edge_attr, graph.node_seg, graph.edge_seg)

    assert embedding.shape == (3, 8)
    assert torch.isfinite(embedding).all()
    torch.testing.assert_close(unpacked, embedding)
    with pytest.raises(ValueError, match="unmatched number of arguments"):
        model(1, 2)


def test_layer_primitives() -> None:
    out = BertLayerNorm(4)(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert out.shape == (1, 4)
    assert abs(float(out.mean().detach())) < 1e-5

    values = _gelu(torch.tensor([-2.0, 0.0, 2.0]))
    assert float(values[1]) == pytest.approx(0.0)
    assert float(values[0]) < float(values[1]) < float(values[2])

    torch.manual_seed(0)
    layer = LinearActivation(4, 2, bias=False)
    assert layer.bias is None
    assert layer(torch.ones((1, 4))).shape == (1, 2)


def test_message_passing_defaults_and_required_arguments() -> None:
    passing = _MessagePassing()
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    assert passing.propagate(edge_index=edge_index, x=torch.tensor([[1.0], [2.0]])).shape == (2, 1)
    with pytest.raises(ValueError, match="propagate requires 'x'"):
        passing.propagate(edge_index=torch.tensor([[0], [1]], dtype=torch.long))
    with pytest.raises(ValueError, match="message requires 'x_j'"):
        _MessagePassing().message()
