"""DrugGNN model."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset as PytorchDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from ...datasets.dataset import DrugResponseDataset, FeatureDataset
from ..drp_model import DRPModel
from ..utils import load_and_select_gene_features


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
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=responses.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        """A single validation step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        """
        drug_graph, cell_features, responses = batch
        outputs = self.model(drug_graph, cell_features)
        loss = self.criterion(outputs, responses)
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=responses.size(0))

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


class _DrugResponsePytorchDataset(PytorchDataset):
    """A PyTorch Dataset to wrap the drug response data for DrugGNN."""

    def __init__(
        self,
        response: np.ndarray,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_features: FeatureDataset,
        drug_features: FeatureDataset,
    ):
        """Initialize the dataset.

        :param response: The drug response values.
        :param cell_line_ids: The cell line IDs.
        :param drug_ids: The drug IDs.
        :param cell_line_features: A FeatureDataset object with cell line features.
        :param drug_features: A FeatureDataset object with drug features.
        """
        self.response = response
        self.cell_line_ids = cell_line_ids
        self.drug_ids = drug_ids

        # preconvert to tensors to avoid per item tensor creation
        self.cell_features = {
            cl_id: torch.tensor(features["gene_expression"], dtype=torch.float32)
            for cl_id, features in cell_line_features.features.items()
        }
        self.response_tensor = torch.tensor(self.response, dtype=torch.float32)

        self.drug_graphs = {
            drug_id: feature_views["drug_graph"] for drug_id, feature_views in drug_features.features.items()
        }

    def __len__(self):
        return len(self.response)

    def __getitem__(self, idx):
        cell_line_id = self.cell_line_ids[idx]
        drug_id = self.drug_ids[idx]

        drug_graph = self.drug_graphs[drug_id]
        cell_feat = self.cell_features[cell_line_id]
        response = self.response_tensor[idx]

        return drug_graph, cell_feat, response


class DrugGNN(DRPModel):
    """DrugGNN model."""

    def __init__(self):
        """Initialize the DrugGNN model."""
        super().__init__()
        self.model: DrugGNNModule | None = None
        self.hyperparameters = {}

    @classmethod
    def get_model_name(cls) -> str:
        """Return the name of the model.

        :return: The name of the model.
        """
        return "DrugGNN"

    @property
    def cell_line_views(self) -> list[str]:
        """Return the sources the model needs as input for describing the cell line.

        :return: The sources the model needs as input for describing the cell line.
        """
        return ["gene_expression"]

    @property
    def drug_views(self) -> list[str]:
        """Return the sources the model needs as input for describing the drug.

        :return: The sources the model needs as input for describing the drug.
        """
        return ["drug_graph"]

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """Build the model.

        :param hyperparameters: The hyperparameters.
        """
        self.hyperparameters = hyperparameters

    def _loader_kwargs(self) -> dict[str, Any]:
        num_workers = int(self.hyperparameters.get("num_workers", 4))
        kw = {
            "num_workers": num_workers,
            "pin_memory": True,
        }
        if num_workers > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = int(self.hyperparameters.get("prefetch_factor", 2))
        return kw

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        **kwargs,
    ):
        """Train the model.

        :param output: The output dataset.
        :param cell_line_input: The cell line input dataset.
        :param drug_input: The drug input dataset.
        :param output_earlystopping: The early stopping output dataset.
        :param kwargs: Additional arguments.
        :raises ValueError: If drug input is not provided.
        """
        if drug_input is None:
            raise ValueError("Drug input is required for DrugGNN")

        # Determine feature sizes
        num_node_features = next(iter(drug_input.features.values()))["drug_graph"].num_node_features
        num_cell_features = next(iter(cell_line_input.features.values()))["gene_expression"].shape[0]

        self.model = DrugGNNModule(
            num_node_features=num_node_features,
            num_cell_features=num_cell_features,
            hidden_dim=self.hyperparameters.get("hidden_dim", 64),
            dropout=self.hyperparameters.get("dropout", 0.2),
            learning_rate=self.hyperparameters.get("learning_rate", 0.001),
        )

        train_dataset = _DrugResponsePytorchDataset(
            response=output.response,
            cell_line_ids=output.cell_line_ids,
            drug_ids=output.drug_ids,
            cell_line_features=cell_line_input,
            drug_features=drug_input,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.hyperparameters.get("batch_size", 1024),
            shuffle=True,
            **self._loader_kwargs(),
        )

        val_loader = None
        if output_earlystopping is not None and len(output_earlystopping) > 0:
            val_dataset = _DrugResponsePytorchDataset(
                response=output_earlystopping.response,
                cell_line_ids=output_earlystopping.cell_line_ids,
                drug_ids=output_earlystopping.drug_ids,
                cell_line_features=cell_line_input,
                drug_features=drug_input,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.hyperparameters.get("batch_size", 32),
                **self._loader_kwargs(),
            )

        trainer = pl.Trainer(
            max_epochs=self.hyperparameters.get("epochs", 100),
            accelerator="auto",
            devices="auto",
            callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5)] if val_loader else None,
            enable_progress_bar=True,
            log_every_n_steps=int(self.hyperparameters.get("log_every_n_steps", 50)),
            precision=self.hyperparameters.get("precision", 32),
        )
        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """Predict drug response.

        :param cell_line_ids: The cell line IDs.
        :param drug_ids: The drug IDs.
        :param cell_line_input: The cell line input dataset.
        :param drug_input: The drug input dataset.
        :raises RuntimeError: If the model has not been trained yet.
        :raises ValueError: If drug input is not provided.
        :return: The predicted drug response.
        """
        if len(drug_ids) == 0 or len(cell_line_ids) == 0:
            print("DrugGNN predict: No  drug or cell line IDs provided; returning empty array.")
            return np.array([])
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        if drug_input is None:
            raise ValueError("Drug input is required for DrugGNN")

        self.model.eval()

        predict_dataset = _DrugResponsePytorchDataset(
            response=np.zeros(len(cell_line_ids)),
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_features=cell_line_input,
            drug_features=drug_input,
        )
        predict_loader = DataLoader(
            predict_dataset,
            batch_size=self.hyperparameters.get("batch_size", 32),
            **self._loader_kwargs(),
        )

        trainer = pl.Trainer(accelerator="auto", devices="auto", enable_progress_bar=False)
        predictions_list = trainer.predict(self.model, dataloaders=predict_loader)

        if not predictions_list:
            print("DrugGNN predict: No predictions were made; returning empty array.")
            return np.array([])

        predictions_flat = [
            item for sublist in predictions_list for item in (sublist if isinstance(sublist, list) else [sublist])
        ]

        predictions = torch.cat(predictions_flat).cpu().numpy()
        return predictions

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Loads the cell line features.

        :param data_path: Path to the gene expression and landmark genes
        :param dataset_name: name of the dataset
        :return: FeatureDataset containing the cell line gene expression features.
        """
        return load_and_select_gene_features(
            feature_type="gene_expression",
            gene_list="landmark_genes_reduced",
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Loads the pre-computed drug graph data.

        :param data_path: Path to the data directory.
        :param dataset_name: Name of the dataset.
        :raises FileNotFoundError: If the drug graph directory is not found.
        :raises ValueError: If no drug graphs are loaded.
        :return: FeatureDataset containing the drug graphs.
        """
        graph_path = Path(data_path) / dataset_name / "drug_graphs"
        if not graph_path.exists():
            raise FileNotFoundError(
                f"Drug graph directory not found at {graph_path}. "
                f"Please run 'create_drug_graphs.py' for the {dataset_name} dataset."
            )

        drug_graphs = {}
        for p_file in graph_path.glob("*.pt"):
            drug_id = p_file.stem
            drug_graphs[drug_id] = torch.load(p_file, weights_only=False)  # noqa: S614

        if not drug_graphs:
            raise ValueError(f"No drug graphs loaded from {graph_path}. Check the directory and file contents.")

        feature_dict = {drug_id: {"drug_graph": graph} for drug_id, graph in drug_graphs.items()}

        return FeatureDataset(features=feature_dict)

    def save_model(self, path: str | Path, drug_name=None):
        """Save the model.

        :param path: The path to save the model to.
        :param drug_name: The name of the drug.
        :raises RuntimeError: If there is no model to save.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        trainer = pl.Trainer()
        trainer.save_checkpoint(path / "model.ckpt", weights_only=True)

        with open(path / "config.json", "w") as f:
            json.dump(self.hyperparameters, f, indent=4)

    def load_model(self, path: str | Path, drug_name=None):
        """Load the model.

        :param path: The path to load the model from.
        :param drug_name: The name of the drug.
        """
        path = Path(path)

        config_path = path / "config.json"
        with open(config_path) as f:
            self.hyperparameters = json.load(f)

        self.model = DrugGNNModule.load_from_checkpoint(
            path / "model.ckpt",
            num_node_features=self.hyperparameters["num_node_features"],
            num_cell_features=self.hyperparameters["num_cell_features"],
            hidden_dim=self.hyperparameters.get("hidden_dim", 64),
            dropout=self.hyperparameters.get("dropout", 0.2),
            learning_rate=self.hyperparameters.get("learning_rate", 0.001),
        )
