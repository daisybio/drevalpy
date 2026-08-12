"""Tests for the DrugGNN graph network and its LightningModule wrapper.

``druggnn/test_predictor.py`` covers the predictor lifecycle end to end; this
file exercises ``algorithm.py`` directly so the network and module contracts are
asserted without a full Lightning fit.
"""

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


def test_graph_net_returns_one_flat_prediction_per_graph() -> None:
    net = _net()

    with torch.no_grad():
        output = net(_drug_graph(n_graphs=3), torch.randn(3, NUM_CELL_FEATURES))

    assert output.shape == (3,)
    assert torch.isfinite(output).all()


def test_graph_net_widens_the_convolution_stack_from_the_hidden_dim() -> None:
    net = _net(hidden_dim=8)

    assert net.conv1.out_channels == 8
    assert net.conv2.out_channels == 16
    assert net.conv3.out_channels == 32
    assert net.drug_embed_fc.in_features == 32


def test_graph_net_combines_equal_width_drug_and_cell_embeddings() -> None:
    net = _net(hidden_dim=8)

    assert net.drug_embed_fc.out_features == 8
    assert net.cell_fc2.out_features == 8
    assert net.combiner_fc1.in_features == 16


def test_graph_net_records_the_dropout_rate_for_functional_use() -> None:
    net = _net(dropout=0.5)

    assert net.dropout == 0.5


def test_graph_net_is_deterministic_in_eval_mode() -> None:
    net = _net(dropout=0.5)
    graph = _drug_graph()
    cell_features = torch.randn(2, NUM_CELL_FEATURES)

    with torch.no_grad():
        first = net(graph, cell_features)
        second = net(graph, cell_features)

    assert torch.allclose(first, second)


def test_graph_net_gradients_reach_the_first_convolution() -> None:
    net = DrugGraphNet(
        num_node_features=NUM_NODE_FEATURES,
        num_cell_features=NUM_CELL_FEATURES,
        hidden_dim=8,
        dropout=0.0,
    )

    net(_drug_graph(), torch.randn(2, NUM_CELL_FEATURES)).sum().backward()

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


def test_module_saves_its_construction_hyperparameters() -> None:
    module = _module(learning_rate=0.005)

    assert module.hparams["num_node_features"] == NUM_NODE_FEATURES
    assert module.hparams["hidden_dim"] == 8
    assert module.hparams["learning_rate"] == pytest.approx(0.005)


def test_module_forward_unpacks_the_three_element_batch() -> None:
    module = _module()
    module.eval()
    batch = (_drug_graph(), torch.randn(2, NUM_CELL_FEATURES), torch.randn(2))

    with torch.no_grad():
        output = module(batch)

    assert output.shape == (2,)


def test_module_training_step_returns_a_scalar_mse_loss() -> None:
    module = _module()
    batch = (_drug_graph(), torch.randn(2, NUM_CELL_FEATURES), torch.randn(2))

    loss = module.training_step(batch, batch_idx=0)

    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_module_validation_step_returns_nothing() -> None:
    module = _module()
    batch = (_drug_graph(), torch.randn(2, NUM_CELL_FEATURES), torch.randn(2))

    assert module.validation_step(batch, batch_idx=0) is None


def test_module_predict_step_matches_forward() -> None:
    module = _module()
    module.eval()
    batch = (_drug_graph(), torch.randn(2, NUM_CELL_FEATURES), torch.randn(2))

    with torch.no_grad():
        assert torch.allclose(module.predict_step(batch, batch_idx=0), module(batch))


def test_module_configures_adam_at_the_requested_learning_rate() -> None:
    module = _module(learning_rate=0.01)

    optimizer = module.configure_optimizers()

    assert isinstance(optimizer, Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
