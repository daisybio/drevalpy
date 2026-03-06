"""PPIGraphGNN model for drug response prediction using PPI networks and GNNExplainer."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset as PytorchDataset
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from ...datasets.dataset import DrugResponseDataset, FeatureDataset
from ..drp_model import DRPModel
from ..lightning_metrics_mixin import RegressionMetricsMixin
from ..utils import load_and_select_gene_features, load_drug_fingerprint_features


class PPIGraphNet(nn.Module):
    """Graph Neural Network for processing PPI networks with gene expression and drug features."""

    def __init__(
        self,
        num_genes: int,
        num_drug_features: int,
        hidden_dim: int = 64,
        num_gnn_layers: int = 3,
        dropout: float = 0.2,
    ):
        """Initialize the network.

        :param num_genes: Number of genes (node features dimension).
        :param num_drug_features: Number of drug features (e.g., fingerprint size).
        :param hidden_dim: The hidden dimension size.
        :param num_gnn_layers: Number of GNN layers.
        :param dropout: The dropout rate.
        """
        super().__init__()
        self.dropout = dropout
        self.num_gnn_layers = num_gnn_layers

        # GNN layers to process PPI graph with gene expression
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GCNConv(1, hidden_dim))
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))

        # Drug encoder (MLP for drug fingerprints)
        self.drug_fc1 = nn.Linear(num_drug_features, hidden_dim)
        self.drug_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Combined prediction layers (PPI graph embedding + drug embedding)
        self.combiner_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.combiner_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_fc = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x, edge_index, batch, drug_features):
        """Forward pass of the network.

        :param x: Node features (gene expression per node).
        :param edge_index: Edge connectivity from PPI network.
        :param batch: Batch assignment vector.
        :param drug_features: Drug fingerprints or other drug features.
        :return: Predicted drug response.
        """
        # Process PPI graph through GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index)
            x = nn.functional.relu(x)
            if i < len(self.gnn_layers) - 1:  # Don't apply dropout after the last GNN layer
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        # Global pooling to get graph-level embedding
        graph_embedding = global_mean_pool(x, batch)

        # Process drug features
        drug_embedding = nn.functional.relu(self.drug_fc1(drug_features))
        drug_embedding = nn.functional.dropout(drug_embedding, p=self.dropout, training=self.training)
        drug_embedding = self.drug_fc2(drug_embedding)

        # Combine graph embedding and drug embedding
        combined = torch.cat([graph_embedding, drug_embedding], dim=1)
        x = nn.functional.relu(self.combiner_fc1(combined))
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = nn.functional.relu(self.combiner_fc2(x))
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        out = self.output_fc(x)
        return out.view(-1)


class PPIGraphGNNModule(RegressionMetricsMixin, pl.LightningModule):
    """The LightningModule for the PPIGraphGNN model."""

    def __init__(
        self,
        num_genes: int,
        num_drug_features: int,
        hidden_dim: int = 64,
        num_gnn_layers: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
    ):
        """Initialize the LightningModule.

        :param num_genes: Number of genes in the gene expression data.
        :param num_drug_features: Number of drug features.
        :param hidden_dim: The hidden dimension size.
        :param num_gnn_layers: Number of GNN layers.
        :param dropout: The dropout rate.
        :param learning_rate: The learning rate.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = PPIGraphNet(
            num_genes=self.hparams["num_genes"],
            num_drug_features=self.hparams["num_drug_features"],
            hidden_dim=self.hparams["hidden_dim"],
            num_gnn_layers=self.hparams["num_gnn_layers"],
            dropout=self.hparams["dropout"],
        )
        self.criterion = nn.MSELoss()

        # Initialize metrics storage for epoch-end R^2 and PCC computation
        self._init_metrics_storage()

    def forward(self, batch):
        """Forward pass of the module.

        :param batch: The batch containing graph data, drug features, and responses.
        :return: The output of the model.
        """
        graph, drug_features, responses = batch
        return self.model(graph.x, graph.edge_index, graph.batch, drug_features)

    def training_step(self, batch, batch_idx):
        """A single training step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        :return: The loss.
        """
        graph, drug_features, responses = batch
        outputs = self.model(graph.x, graph.edge_index, graph.batch, drug_features)
        loss = self.criterion(outputs, responses)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=responses.size(0))

        # Store predictions and targets for epoch-end metrics via mixin
        self._store_predictions(outputs, responses, is_training=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """A single validation step.

        :param batch: The batch.
        :param batch_idx: The batch index.
        """
        graph, drug_features, responses = batch
        outputs = self.model(graph.x, graph.edge_index, graph.batch, drug_features)
        loss = self.criterion(outputs, responses)
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=responses.size(0))

        # Store predictions and targets for epoch-end metrics via mixin
        self._store_predictions(outputs, responses, is_training=False)

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


class _PPIGraphDataset(PytorchDataset):
    """A PyTorch Dataset to wrap PPI graphs with gene expression and drug features."""

    def __init__(
        self,
        response: np.ndarray,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_features: FeatureDataset,
        drug_features: FeatureDataset,
        ppi_graph_template: Data,
    ):
        """Initialize the dataset.

        :param response: The drug response values.
        :param cell_line_ids: The cell line IDs.
        :param drug_ids: The drug IDs.
        :param cell_line_features: A FeatureDataset object with cell line gene expression features.
        :param drug_features: A FeatureDataset object with drug features (fingerprints).
        :param ppi_graph_template: Template PPI graph (same structure for all samples).
        """
        self.response = response
        self.cell_line_ids = cell_line_ids
        self.drug_ids = drug_ids
        self.ppi_graph_template = ppi_graph_template

        # Preconvert gene expression to tensors
        self.cell_features = {
            cl_id: torch.tensor(features["gene_expression"], dtype=torch.float32)
            for cl_id, features in cell_line_features.features.items()
        }

        # Preconvert drug features to tensors
        self.drug_features = {
            drug_id: torch.tensor(features["fingerprints"], dtype=torch.float32)
            for drug_id, features in drug_features.features.items()
        }

        self.response_tensor = torch.tensor(self.response, dtype=torch.float32)

    def __len__(self):
        return len(self.response)

    def __getitem__(self, idx):
        cell_line_id = self.cell_line_ids[idx]
        drug_id = self.drug_ids[idx]

        # Create a copy of the PPI graph and set node features to gene expression
        graph = self.ppi_graph_template.clone()
        gene_expr = self.cell_features[cell_line_id]

        # Set node features as gene expression values (expand dims to match expected shape)
        graph.x = gene_expr.unsqueeze(1)

        # Get drug features
        drug_feat = self.drug_features[drug_id]

        response = self.response_tensor[idx]

        return graph, drug_feat, response


class PPIGraphGNN(DRPModel):
    """PPIGraphGNN model using PPI networks and gene expression with GNNExplainer support."""

    def __init__(self):
        """Initialize the PPIGraphGNN model."""
        super().__init__()
        self.model: PPIGraphGNNModule | None = None
        self.hyperparameters = {}
        self.ppi_graph_template: Data | None = None
        self.explainer: Explainer | None = None

    @classmethod
    def get_model_name(cls) -> str:
        """Return the name of the model.

        :return: The name of the model.
        """
        return "PPIGraphGNN"

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
        return ["fingerprints"]

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """Build the model.

        :param hyperparameters: The hyperparameters.
        """
        # Log hyperparameters to wandb if enabled
        self.log_hyperparameters(hyperparameters)

        self.hyperparameters = hyperparameters

    def _validate_gene_order(self, cell_line_input: FeatureDataset) -> None:
        """
        Validate that the gene order in the PPI graph matches the gene expression feature order.

        :param cell_line_input: FeatureDataset with gene expression features
        :raises ValueError: If gene order doesn't match or validation fails
        :raises RuntimeError: If PPI graph template is not loaded
        """
        if self.ppi_graph_template is None:
            raise RuntimeError("PPI graph template not loaded")

        # Check if the PPI graph has gene_names attributes
        if not hasattr(self.ppi_graph_template, "gene_names"):
            raise ValueError(
                "PPI graph doesn't contain gene_names metadata. "
                "Please regenerate the PPI graph using the updated create_ppi_graphs.py script."
            )

        ppi_gene_names = self.ppi_graph_template.gene_names

        # Get gene names from cell_line_input meta_info
        if "gene_expression" not in cell_line_input.meta_info:
            raise ValueError("cell_line_input doesn't contain gene_expression meta_info")

        expr_gene_names = list(cell_line_input.meta_info["gene_expression"])

        # Validate number of genes matches
        if len(ppi_gene_names) != len(expr_gene_names):
            raise ValueError(
                f"Gene count mismatch: PPI graph has {len(ppi_gene_names)} genes, "
                f"but gene expression has {len(expr_gene_names)} genes. "
                f"Ensure both use the same gene list (e.g., landmark_genes_reduced)."
            )

        # Validate gene order matches
        for i, (ppi_gene, expr_gene) in enumerate(zip(ppi_gene_names, expr_gene_names, strict=False)):
            if ppi_gene != expr_gene:
                raise ValueError(
                    f"Gene order mismatch at position {i}: "
                    f"PPI graph has '{ppi_gene}' but gene expression has '{expr_gene}'. "
                    f"Regenerate PPI graph using: python -m drevalpy.datasets.featurizer.create_ppi_graphs"
                )

        print(f"✓ Validated: PPI graph and gene expression have matching gene order ({len(ppi_gene_names)} genes)")

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
        :param drug_input: The drug input dataset (fingerprints).
        :param output_earlystopping: The early stopping output dataset.
        :param kwargs: Additional arguments.
        :raises RuntimeError: If PPI graph template is not loaded.
        :raises ValueError: If drug_input is not provided.
        :raises ValueError: If gene order doesn't match between PPI graph and gene expression.
        """
        if self.ppi_graph_template is None:
            raise RuntimeError("PPI graph template not loaded. Call load_drug_features() first.")

        if drug_input is None:
            raise ValueError("drug_input (fingerprints) is required for PPIGraphGNN.")

        # Validate gene order consistency
        self._validate_gene_order(cell_line_input)

        # Determine feature sizes
        num_drug_features = next(iter(drug_input.features.values()))["fingerprints"].shape[0]

        self.model = PPIGraphGNNModule(
            num_genes=1,
            num_drug_features=num_drug_features,
            hidden_dim=self.hyperparameters.get("hidden_dim", 64),
            num_gnn_layers=self.hyperparameters.get("num_gnn_layers", 3),
            dropout=self.hyperparameters.get("dropout", 0.2),
            learning_rate=self.hyperparameters.get("learning_rate", 0.001),
        )

        # Initialize GNNExplainer
        self.explainer = Explainer(
            model=self.model.model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="regression",
                task_level="graph",
                return_type="raw",
            ),
        )

        train_dataset = _PPIGraphDataset(
            response=output.response,
            cell_line_ids=output.cell_line_ids,
            drug_ids=output.drug_ids,
            cell_line_features=cell_line_input,
            drug_features=drug_input,
            ppi_graph_template=self.ppi_graph_template,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.hyperparameters.get("batch_size", 32),
            shuffle=True,
            **self._loader_kwargs(),
        )

        val_loader = None
        if output_earlystopping is not None and len(output_earlystopping) > 0:
            val_dataset = _PPIGraphDataset(
                response=output_earlystopping.response,
                cell_line_ids=output_earlystopping.cell_line_ids,
                drug_ids=output_earlystopping.drug_ids,
                cell_line_features=cell_line_input,
                drug_features=drug_input,
                ppi_graph_template=self.ppi_graph_template,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.hyperparameters.get("batch_size", 32),
                **self._loader_kwargs(),
            )

        # Set up wandb logger if project is provided
        loggers = []
        if self.wandb_project is not None:
            from pytorch_lightning.loggers import WandbLogger

            logger = WandbLogger(project=self.wandb_project, log_model=False)
            loggers.append(logger)

        trainer = pl.Trainer(
            max_epochs=self.hyperparameters.get("epochs", 100),
            accelerator="auto",
            devices="auto",
            callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5)] if val_loader else None,
            logger=loggers if loggers else True,
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
        :param drug_input: The drug input dataset (fingerprints).
        :raises RuntimeError: If the model has not been trained yet.
        :raises RuntimeError: If PPI graph template is not loaded.
        :raises ValueError: If drug_input is not provided.
        :return: The predicted drug response.
        """
        if len(cell_line_ids) == 0:
            print("PPIGraphGNN predict: No cell line IDs provided; returning empty array.")
            return np.array([])
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        if self.ppi_graph_template is None:
            raise RuntimeError("PPI graph template not loaded.")
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) is required for PPIGraphGNN.")

        self.model.eval()

        predict_dataset = _PPIGraphDataset(
            response=np.zeros(len(cell_line_ids)),
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_features=cell_line_input,
            drug_features=drug_input,
            ppi_graph_template=self.ppi_graph_template,
        )
        predict_loader = DataLoader(
            predict_dataset,
            batch_size=self.hyperparameters.get("batch_size", 32),
            **self._loader_kwargs(),
        )

        trainer = pl.Trainer(accelerator="auto", devices="auto", enable_progress_bar=False)
        predictions_list = trainer.predict(self.model, dataloaders=predict_loader)

        if not predictions_list:
            print("PPIGraphGNN predict: No predictions were made; returning empty array.")
            return np.array([])

        predictions_flat = [
            item for sublist in predictions_list for item in (sublist if isinstance(sublist, list) else [sublist])
        ]

        predictions = torch.cat(predictions_flat).cpu().numpy()
        return predictions

    def explain(
        self,
        cell_line_id: str,
        drug_id: str,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset,
        top_k_edges: int = 20,
    ) -> dict[str, Any]:
        """
        Use GNNExplainer to extract important subnetwork for a specific cell line-drug pair.

        :param cell_line_id: The cell line ID to explain.
        :param drug_id: The drug ID to explain.
        :param cell_line_input: The cell line input dataset.
        :param drug_input: The drug input dataset.
        :param top_k_edges: Number of top important edges to return.
        :raises RuntimeError: If model or explainer is not initialized.
        :return: Dictionary containing explanation with important edges and nodes.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized. Train the model first.")
        if self.ppi_graph_template is None:
            raise RuntimeError("PPI graph template not loaded.")

        self.model.eval()

        # Create graph with gene expression for this cell line
        graph = self.ppi_graph_template.clone()
        gene_expr = torch.tensor(
            cell_line_input.features[cell_line_id]["gene_expression"], dtype=torch.float32
        ).unsqueeze(1)
        graph.x = gene_expr

        # Get drug features
        drug_features = torch.tensor(drug_input.features[drug_id]["fingerprints"], dtype=torch.float32).unsqueeze(0)

        # Get explanation
        with torch.no_grad():
            explanation = self.explainer(
                x=graph.x,
                edge_index=graph.edge_index,
                batch=torch.zeros(graph.num_nodes, dtype=torch.long),
                drug_features=drug_features,
            )

        # Extract important edges
        edge_mask = explanation.edge_mask.cpu().numpy()
        edge_index = graph.edge_index.cpu().numpy()

        # Get top-k edges
        top_edge_indices = np.argsort(edge_mask)[::-1][:top_k_edges]
        important_edges = [(int(edge_index[0, i]), int(edge_index[1, i])) for i in top_edge_indices]
        edge_scores = [float(edge_mask[i]) for i in top_edge_indices]

        # Get gene names if available
        gene_names = getattr(self.ppi_graph_template, "gene_names", None)
        if gene_names is not None:
            important_edges_with_names = [
                (gene_names[src], gene_names[dst], score)
                for (src, dst), score in zip(important_edges, edge_scores, strict=True)
            ]
        else:
            important_edges_with_names = [
                (src, dst, score) for (src, dst), score in zip(important_edges, edge_scores, strict=True)
            ]

        return {
            "cell_line_id": cell_line_id,
            "drug_id": drug_id,
            "important_edges": important_edges_with_names,
            "edge_mask": edge_mask,
            "explanation": explanation,
        }

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Loads the cell line features.

        :param data_path: Path to the gene expression
        :param dataset_name: name of the dataset
        :raises FileNotFoundError: If PPI graph is not found at the specified path.
        :return: FeatureDataset containing the cell line gene expression features.
        """
        # Load PPI graph first
        ppi_graph_path = Path(data_path) / dataset_name / "ppi_graph.pt"
        if not ppi_graph_path.exists():
            raise FileNotFoundError(
                f"PPI graph not found at {ppi_graph_path}. "
                f"Please run 'python -m drevalpy.datasets.featurizer.create_ppi_graphs {dataset_name}' first."
            )

        self.ppi_graph_template = torch.load(ppi_graph_path, weights_only=False)  # noqa: S614
        print(
            f"Loaded PPI graph with {self.ppi_graph_template.num_nodes} nodes "
            f"and {self.ppi_graph_template.num_edges} edges"
        )

        return load_and_select_gene_features(
            feature_type="gene_expression",
            gene_list=None,
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Loads the drug features (fingerprints) and PPI graph.

        :param data_path: Path to the data directory.
        :param dataset_name: Name of the dataset.
        :return: FeatureDataset containing drug fingerprints.
        """
        # Load drug fingerprints
        return load_drug_fingerprint_features(data_path, dataset_name, fill_na=True)

    def save(self, directory: str) -> None:
        """
        Save the trained model, hyperparameters, and gene expression scaler to the given directory.

        This enables full reconstruction of the model using `load`.

        Files saved:
        - model.pt: PyTorch state_dict of the trained model
        - hyperparameters.json: Dictionary containing all relevant model hyperparameters
        - ppi_graph.pt: PPI graph template

        :param directory: Target directory to store all model artifacts
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path / "model.pt")  # noqa: S614

        with open(path / "hyperparameters.json", "w") as f:
            json.dump(self.hyperparameters, f)

        torch.save(self.ppi_graph_template, path / "ppi_graph.pt")

    @classmethod
    def load(cls, directory: str) -> "PPIGraphGNN":
        """
        Load a trained PPI Graph GNN model from the given directory.

        This includes:
        - model.pt: PyTorch state_dict of the trained model
        - hyperparameters.json: Dictionary containing all relevant model hyperparameters
        - ppi_graph.pt: PPI graph template

        :param directory: The path to load the model from.
        :return: The loaded PPIGraphGNN model.
        :raises FileNotFoundError: If any of the required files are not found.
        """
        path = Path(directory)

        hpam_path = path / "hyperparameters.json"
        model_file = path / "model.pt"
        ppi_graph_path = path / "ppi_graph.pt"
        if not hpam_path.exists() or not model_file.exists() or not ppi_graph_path.exists():
            raise FileNotFoundError(
                f"Missing required files in {directory}. " f"Please make sure all files are present and try again."
            )

        instance = cls()

        with open(hpam_path) as f:
            instance.hyperparameters = json.load(f)

        instance.ppi_graph_template = torch.load(ppi_graph_path, weights_only=False)  # noqa: S614

        instance.model = PPIGraphGNNModule(
            num_genes=instance.hyperparameters["num_genes"],
            num_drug_features=instance.hyperparameters["num_drug_features"],
            hidden_dim=instance.hyperparameters.get("hidden_dim", 64),
            num_gnn_layers=instance.hyperparameters.get("num_gnn_layers", 3),
            dropout=instance.hyperparameters.get("dropout", 0.2),
            learning_rate=instance.hyperparameters.get("learning_rate", 0.001),
        )
        instance.model.load_state_dict(torch.load(model_file, weights_only=True))
        instance.model.eval()

        # Reinitialize explainer
        instance.explainer = Explainer(
            model=instance.model.model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="regression",
                task_level="graph",
                return_type="raw",
            ),
        )
        return instance
