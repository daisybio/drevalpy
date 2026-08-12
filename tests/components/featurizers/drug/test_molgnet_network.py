"""Tests for the MolGNet graph conversion and network.

Mirrors :mod:`drevalpy.components.featurizers.drug._molgnet_network` (underscore
stripped, per the AGENTS.md private-module rule). The module was at zero coverage
because nothing imported it outside the checkpoint path; nothing here needs the
checkpoint, only a randomly initialised, deliberately tiny ``MolGNet``.
"""

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


def test_cumulative_offsets_are_monotonic() -> None:
    assert np.all(np.diff(atom_cumsum) > 0)
    assert np.all(np.diff(bond_cumsum) > 0)


def test_mol_to_graph_encodes_eight_atom_features() -> None:
    graph = _graph("CCO")

    assert graph.x.shape == (3, 8)
    assert graph.x.dtype == torch.long


def test_mol_to_graph_emits_both_bond_directions() -> None:
    graph = _graph("CCO")

    assert graph.edge_index.shape == (2, 4)
    assert graph.edge_attr.shape == (4, 5)


def test_mol_to_graph_emits_empty_edges_for_a_single_atom() -> None:
    graph = _graph("C")

    assert graph.x.shape == (1, 8)
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, 5)


def test_mol_to_graph_handles_aromatic_rings() -> None:
    graph = _graph("c1ccccc1")

    assert graph.x.shape == (6, 8)
    assert graph.edge_index.shape == (2, 12)


def test_mol_to_graph_rejects_none() -> None:
    with pytest.raises(ValueError, match="must not be None"):
        mol_to_graph_data_obj_complex(None)


def test_self_loop_adds_one_edge_per_node() -> None:
    graph = _graph("CCO")
    before_edges = graph.edge_index.shape[1]

    looped = SelfLoop()(graph)

    assert looped.edge_index.shape[1] == before_edges + 3
    assert looped.edge_attr.shape[0] == before_edges + 3


def test_add_seg_id_attaches_zero_filled_segment_tensors() -> None:
    graph = AddSegId()(SelfLoop()(_graph("CCO")))

    assert graph.node_seg.tolist() == [0, 0, 0]
    assert graph.edge_seg.shape[0] == graph.num_edges
    assert torch.all(graph.edge_seg == 0)


def test_add_seg_id_rejects_a_graph_without_a_node_count() -> None:
    class _NodelessGraph:
        num_nodes = None
        num_edges = 0

    with pytest.raises(ValueError, match="graph reports no node count"):
        AddSegId()(_NodelessGraph())


def test_bert_layer_norm_normalizes_the_last_dimension() -> None:
    layer = BertLayerNorm(4)

    out = layer(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    assert out.shape == (1, 4)
    assert abs(float(out.mean().detach())) < 1e-5


def test_molgnet_forward_returns_one_embedding_row_per_atom() -> None:
    torch.manual_seed(0)
    graph = AddSegId()(SelfLoop()(_graph("CCO")))
    model = MolGNet(num_layer=1, emb_dim=8, heads=2, num_message_passing=1, drop_ratio=0.0)
    model.eval()

    with torch.no_grad():
        embedding = model(graph)

    assert embedding.shape == (3, 8)
    assert torch.isfinite(embedding).all()


def test_molgnet_forward_accepts_unpacked_tensors() -> None:
    torch.manual_seed(0)
    graph = AddSegId()(SelfLoop()(_graph("CCO")))
    model = MolGNet(num_layer=1, emb_dim=8, heads=2, num_message_passing=1, drop_ratio=0.0)
    model.eval()

    with torch.no_grad():
        embedding = model(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.node_seg,
            graph.edge_seg,
        )

    assert embedding.shape == (3, 8)


def test_molgnet_forward_rejects_an_unexpected_argument_count() -> None:
    model = MolGNet(num_layer=1, emb_dim=8, heads=2, num_message_passing=1, drop_ratio=0.0)

    with pytest.raises(ValueError, match="unmatched number of arguments"):
        model(1, 2)


def test_gelu_is_zero_at_the_origin_and_monotonic() -> None:
    values = _gelu(torch.tensor([-2.0, 0.0, 2.0]))

    assert float(values[1]) == pytest.approx(0.0)
    assert float(values[0]) < float(values[1]) < float(values[2])


def test_linear_activation_without_a_bias_uses_the_plain_gelu_path() -> None:
    torch.manual_seed(0)
    layer = LinearActivation(4, 2, bias=False)

    out = layer(torch.ones((1, 4)))

    assert layer.bias is None
    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()


def test_message_passing_defaults_forward_messages_unchanged() -> None:
    passing = _MessagePassing()
    x = torch.tensor([[1.0], [2.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    out = passing.propagate(edge_index=edge_index, x=x)

    assert out.shape == (2, 1)


def test_message_passing_requires_node_features() -> None:
    passing = _MessagePassing()

    with pytest.raises(ValueError, match="propagate requires 'x'"):
        passing.propagate(edge_index=torch.tensor([[0], [1]], dtype=torch.long))


def test_message_passing_message_requires_source_features() -> None:
    with pytest.raises(ValueError, match="message requires 'x_j'"):
        _MessagePassing().message()
