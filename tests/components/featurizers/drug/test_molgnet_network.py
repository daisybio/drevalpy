"""Tests for the MolGNet graph conversion and (randomly initialised, tiny) network."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from rdkit import Chem

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


def _graph(smiles: str = "CCO"):
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
    graph = _graph(smiles)

    assert graph.x.shape == (nodes, 8)
    assert graph.x.dtype == torch.long
    assert graph.edge_index.shape == (2, edges)
    assert graph.edge_attr.shape == (edges, 5)


def test_mol_to_graph_rejects_none() -> None:
    with pytest.raises(ValueError, match="must not be None"):
        mol_to_graph_data_obj_complex(None)


def test_self_loop_and_seg_id_transforms() -> None:
    graph = _graph("CCO")
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
    graph = AddSegId()(SelfLoop()(_graph("CCO")))
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
