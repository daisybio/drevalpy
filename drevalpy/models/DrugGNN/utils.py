"""DrugGNN model utils."""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class DrugGraphNet(nn.Module):
    """Neural network for DrugGNN."""

    def __init__(self, num_node_features, num_cell_features, hidden_dim=64, dropout=0.2):
        """Initialize the network.

        :param num_node_features: Number of features for each node in the drug graph.
        :param num_cell_features: Number of features for the cell line.
        :param hidden_dim: The hidden dimension size.
        :param dropout: The dropout rate.
        """
        super().__init__()
        self.dropout = dropout

        # Drug Encoder (GNN)
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim * 4)
        self.drug_embed_fc = nn.Linear(hidden_dim * 4, hidden_dim)

        # Cell Line Encoder (MLP)
        self.cell_fc1 = nn.Linear(num_cell_features, hidden_dim * 2)
        self.cell_fc2 = nn.Linear(hidden_dim * 2, hidden_dim)

        # Combiner and Regressor
        self.combiner_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.combiner_fc2 = nn.Linear(hidden_dim, 32)
        self.output_fc = nn.Linear(32, 1)

    def forward(self, drug_graph, cell_features):
        """Forward pass of the network.

        :param drug_graph: The drug graph.
        :param cell_features: The cell line features.
        :return: The output of the network.
        """
        # Process drug graph
        x, edge_index, batch = drug_graph.x, drug_graph.edge_index, drug_graph.batch

        x = self.conv1(x, edge_index)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = nn.functional.relu(x)

        drug_embedding = global_mean_pool(x, batch)
        drug_embedding = self.drug_embed_fc(drug_embedding)

        # Process cell line features
        cell_embedding = nn.functional.relu(self.cell_fc1(cell_features))
        cell_embedding = nn.functional.dropout(cell_embedding, p=self.dropout, training=self.training)
        cell_embedding = self.cell_fc2(cell_embedding)

        # Concatenate and predict
        combined = torch.cat([drug_embedding, cell_embedding], dim=1)
        x = nn.functional.relu(self.combiner_fc1(combined))
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = nn.functional.relu(self.combiner_fc2(x))
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        out = self.output_fc(x)
        return out.view(-1)
