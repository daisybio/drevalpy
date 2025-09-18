"""DrugGNN model utils."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
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


class DrugGNNModule(pl.LightningModule):
    """The LightningModule for the DrugGNN model."""

    def __init__(
        self,
        num_node_features: int,
        num_cell_features: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
    ):
        """Initialize the LightningModule.

        :param num_node_features: Number of features for each node in the drug graph.
        :param num_cell_features: Number of features for the cell line.
        :param hidden_dim: The hidden dimension size.
        :param dropout: The dropout rate.
        :param learning_rate: The learning rate.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = DrugGraphNet(
            num_node_features=self.hparams["num_node_features"],
            num_cell_features=self.hparams["num_cell_features"],
            hidden_dim=self.hparams["hidden_dim"],
            dropout=self.hparams["dropout"],
        )
        self.criterion = nn.MSELoss()

    def forward(self, batch):
        """Forward pass of the module.

        :param batch: The batch.
        :return: The output of the model.
        """
        drug_graph, cell_features, _ = batch
        return self.model(drug_graph, cell_features)

    def training_step(self, batch, batch_idx):
        """A single training step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        :return: The loss.
        """
        drug_graph, cell_features, responses = batch
        outputs = self.model(drug_graph, cell_features)
        loss = self.criterion(outputs, responses)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """A single validation step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        """
        drug_graph, cell_features, responses = batch
        outputs = self.model(drug_graph, cell_features)
        loss = self.criterion(outputs, responses)
        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """A single prediction step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        :param dataloader_idx: The dataloader index.
        :return: The output of the model.
        """
        return self.forward(batch)

    def configure_optimizers(self):
        """Configure the optimizer.

        :return: The optimizer.
        """
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
