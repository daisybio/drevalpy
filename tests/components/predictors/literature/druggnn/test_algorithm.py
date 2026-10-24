"""Tests for the DrugGNN graph network and its LightningModule wrapper."""

from __future__ import annotations

import pytest
import torch
from torch.optim import Adam
from torch_geometric.data import Data

from drevalpy.components.predictors.literature.druggnn.algorithm import DrugGNNModule, DrugGraphNet

NUM_NODE_FEATURES = 5
NUM_CELL_FEATURES = 7


def _drug_graph(*, n_graphs: int = 2, nodes_per_graph: int = 3) -> Data:
    n_nodes = n_graphs * nodes_per_graph
    edges = [
        [node, node + 1]
        for graph in range(n_graphs)
        for node in range(graph * nodes_per_graph, (graph + 1) * nodes_per_graph - 1)
    ]
    return Data(
        x=torch.randn(n_nodes, NUM_NODE_FEATURES),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        batch=torch.arange(n_graphs).repeat_interleave(nodes_per_graph),
    )


def _net(*, hidden_dim: int = 8, dropout: float = 0.0) -> DrugGraphNet:
    net = DrugGraphNet(
        num_node_features=NUM_NODE_FEATURES,
        num_cell_features=NUM_CELL_FEATURES,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    net.eval()
    return net


def test_graph_net_shapes_widths_and_gradients() -> None:
    net = _net(hidden_dim=8, dropout=0.5)
    graph = _drug_graph(n_graphs=3)
    cells = torch.randn(3, NUM_CELL_FEATURES)

    with torch.no_grad():
        first = net(graph, cells)
        second = net(graph, cells)

    assert first.shape == (3,)
    assert torch.isfinite(first).all()
    assert torch.allclose(first, second)
    assert net.dropout == 0.5
    assert [net.conv1.out_channels, net.conv2.out_channels, net.conv3.out_channels] == [8, 16, 32]
    assert net.drug_embed_fc.in_features == 32
    assert (net.drug_embed_fc.out_features, net.cell_fc2.out_features, net.combiner_fc1.in_features) == (8, 8, 16)

    net.train()
    net(graph, cells).sum().backward()
    assert net.conv1.lin.weight.grad is not None


def _module(**overrides: object) -> DrugGNNModule:
    kwargs: dict[str, object] = {
        "num_node_features": NUM_NODE_FEATURES,
        "num_cell_features": NUM_CELL_FEATURES,
        "hidden_dim": 8,
        "dropout": 0.0,
    }
    kwargs.update(overrides)
    return DrugGNNModule(**kwargs)  # type: ignore[arg-type]


def test_module_hyperparameters_forward_steps_and_optimizer() -> None:
    module = _module(learning_rate=0.01)
    batch = (_drug_graph(), torch.randn(2, NUM_CELL_FEATURES), torch.randn(2))
    assert module.hparams["num_node_features"] == NUM_NODE_FEATURES
    assert module.hparams["hidden_dim"] == 8
    assert module.hparams["learning_rate"] == pytest.approx(0.01)

    loss = module.training_step(batch, batch_idx=0)
    assert loss.ndim == 0
    assert loss.item() >= 0.0
    assert module.validation_step(batch, batch_idx=0) is None

    module.eval()
    with torch.no_grad():
        output = module(batch)
        assert output.shape == (2,)
        assert torch.allclose(module.predict_step(batch, batch_idx=0), output)

    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
