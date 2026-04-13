"""XGDP model."""

import json
import os
import secrets
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytorch_lightning as pl
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset as PytorchDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from ...datasets.dataset import DrugResponseDataset, FeatureDataset
from ..drp_model import DRPModel
from ..lightning_metrics_mixin import RegressionMetricsMixin
from ..utils import load_and_select_gene_features

from .model_utils import XGDPPredictor

class XGDP(DRPModel):
    """XGDP model for ..."""

    cell_line_views = ["gene_expression"]
    drug_views = ["drug_graph"]
    early_stopping = True

    def __init__(self) -> None:
        """Initialize the XGDP model."""
        super().__init__()
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: XGDPPredictor | None = None
        self.hyperparameters: dict[str, Any] = {}
        self.gene_expression_scaler: StandardScaler | None = None
        self.gene_expression_normalizer: MinMaxScaler | None = None

    @classmethod
    def get_model_name(cls) -> str:
        """
        Get the model name.

        :returns: XGDP
        """
        return "XGDP"
    
    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Builds the XGDP model with the specified hyperparameters.

        :param hyperparameters: TODO: ADD HYPERPARAMETERS
        """

        self.hyperparameters = hyperparameters

        self.model = XGDPPredictor(
            name_hyperparameter = hyperparameter["name_hyperparameter"]
        )

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Loads the cell line features.

        :param data_path: Path to the gene expression and landmark genes
        :param dataset_name: name of the dataset
        :return: FeatureDataset containing the cell line gene expression features.
        """
        return load_and_select_gene_features(
            feature_type="gene_expression",
            gene_list="landmark_genes",
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
            drug_graphs[drug_id] = torch.load(p_file, weights_only=False)

        if not drug_graphs:
            raise ValueError(f"No drug graphs loaded from {graph_path}. Check the directory and file contents.")

        feature_dict = {drug_id: {"drug_graph": graph} for drug_id, graph in drug_graphs.items()}

        return FeatureDataset(features=feature_dict)
