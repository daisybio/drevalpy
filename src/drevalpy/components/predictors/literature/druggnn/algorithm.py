"""DrugGNN neural network components."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool


class DrugGraphNet(nn.Module):
    """Neural network for DrugGNN."""

    def __init__(
        self, num_node_features: int, num_cell_features: int, hidden_dim: int = 64, dropout: float = 0.2
    ) -> None:
        """Initialize the network.

        :param num_node_features: Number of features for each node in the drug graph.
        :param num_cell_features: Number of features for the sample.
        :param hidden_dim: Width of the first graph convolution and of the drug and sample embeddings; later layers
            scale from it.
        :param dropout: Dropout probability between layers.
        """
        super().__init__()
        self.dropout = dropout

        # Drug Encoder (GNN)
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim * 4)
        self.drug_embed_fc = nn.Linear(hidden_dim * 4, hidden_dim)

        # Sample encoder (MLP)
        self.cell_fc1 = nn.Linear(num_cell_features, hidden_dim * 2)
        self.cell_fc2 = nn.Linear(hidden_dim * 2, hidden_dim)

        # Combiner and Regressor
        self.combiner_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.combiner_fc2 = nn.Linear(hidden_dim, 32)
        self.output_fc = nn.Linear(32, 1)

    def forward(self, drug_graph: Batch, cell_features: torch.Tensor) -> torch.Tensor:
        """Predict one response per pair from a drug graph and sample features.

        :param drug_graph: Batched drug graphs, one graph per pair.
        :param cell_features: Sample feature matrix of shape [n_pairs, num_cell_features].
        :returns: Predicted responses of shape [n_pairs].
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

        # Process sample features
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
    ) -> None:
        """Initialize the LightningModule.

        :param num_node_features: Number of features for each node in the drug graph.
        :param num_cell_features: Number of features for the sample.
        :param hidden_dim: Hidden dimension passed to :class:`DrugGraphNet`.
        :param dropout: Dropout probability passed to :class:`DrugGraphNet`.
        :param learning_rate: Learning rate of the Adam optimizer.
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

    def forward(self, batch: tuple[Batch, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Predict responses for a loader batch.

        :param batch: Tuple of batched drug graphs, sample features and responses; the responses are ignored.
        :returns: Predicted responses of shape [n_pairs].
        """
        drug_graph, cell_features, _ = batch
        return self.model(drug_graph, cell_features)

    def training_step(self, batch: tuple[Batch, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute and log the MSE training loss of one batch.

        :param batch: Tuple of batched drug graphs, sample features and responses.
        :param batch_idx: Index of the batch within the epoch; unused.
        :returns: The MSE loss.
        """
        drug_graph, cell_features, responses = batch
        outputs = self.model(drug_graph, cell_features)
        loss = self.criterion(outputs, responses)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=responses.size(0))

        return loss

    def validation_step(self, batch: tuple[Batch, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Compute and log the MSE validation loss of one batch.

        :param batch: Tuple of batched drug graphs, sample features and responses.
        :param batch_idx: Index of the batch within the epoch; unused.
        """
        drug_graph, cell_features, responses = batch
        outputs = self.model(drug_graph, cell_features)
        loss = self.criterion(outputs, responses)
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=responses.size(0))

    def predict_step(
        self, batch: tuple[Batch, torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Predict responses for one batch.

        :param batch: Tuple of batched drug graphs, sample features and responses.
        :param batch_idx: Index of the batch within the loader; unused.
        :param dataloader_idx: Index of the dataloader; unused.
        :returns: Predicted responses of shape [n_pairs].
        """
        return self.forward(batch)

    def configure_optimizers(self) -> Adam:
        """Create the optimizer.

        :returns: An Adam optimizer over all parameters.
        """
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
