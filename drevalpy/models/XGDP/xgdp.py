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
import torch_geometric
from torch.optim import Adam
from torch.utils.data import Dataset as PytorchDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from ...datasets.dataset import DrugResponseDataset, FeatureDataset
from ..drp_model import DRPModel
from ..lightning_metrics_mixin import RegressionMetricsMixin
from ..utils import load_and_select_gene_features

from .model_utils import XGDPPredictor

import pandas as pd
import numpy as np
import sys 
import torch.nn as nn
import tqdm
import time

import models
import utils as u

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
        '''
         # Log hyperparameters to wandb if enabled
        self.log_hyperparameters(hyperparameters)

        self.hyperparameters = hyperparameters
        # init model in train
        '''

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
            raise ValueError("Drug input is required for XGDP")
        
        #step1 load data

        #step 2: preprocess data

        #step 4: init model + train parameters
            #to do: get feature size from data? -> 
        with_attention = []
        model_name = self.hyperparameters["model"]
        model_class = getattr(models,model_name )
        #is done after cv split
        #self.model = model_class() #check hyper parameter for each model
        if 'GAT' in model_name:
            return_attention_weights = True
        else:
            return_attention_weights = False
        n_epochs = self.hyperparameters["n_epochs"]
        lr = self.hyperparameters["lr"]
        train_batch_size = self.hyperparameters["train_batch_size"]
        test_batch_size = self.hyperparameters["test_batch_size"]
        val_batch_size = self.hyperparameters["val_batch_size"]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #step 3: build data loaders + split data
        test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
        train_data = Subset(data,train_index)
        train_data = Subset(data,val_index)
        train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=False)
        val_loader = DataLoader(test_data, batch_size=val_batch_size, shuffle=False)
        #step 5: train loop

        #creates model for each cv split
        self.model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_losses = []
        val_losses = []
        val_pearsons = []
        best_ret = []
        for epoch in tqdm(range(n_epochs)):
            start_time = time.time()
            train_loss = train(model, device, train_loader, optimizer, epoch+1, log_interval)
            G,P = predicting(model, device, val_loader)
            ret = [u.rmse(G,P),u.mse(G,P),u.pearson(G,P),u.spearman(G,P),coeffi_determ(G,P)]
            train_losses.append(train_loss)
            val_losses.append(ret[1])
            val_pearsons.append(ret[2])

        self.model = best_model
        #step 6: save mdoel (maybe not neccecary beacuse class)


class XGDP(DRPModel):
    #to do
    def __init__(self):
        """Initialize the DrugGNN model."""
        pass

    @classmethod
    def get_model_name(cls) -> str:
        """Return the name of the model.

        :return: The name of the model.
        """
        return "XGDP"

    @property
    def cell_line_views(self) -> list[str]:
        """Return the sources the model needs as input for describing the cell line.

        :return: The sources the model needs as input for describing the cell line.
        """
        return ["gene_expression"]
    
    #to do
    @property
    def drug_views(self) -> list[str]:
        """Return the sources the model needs as input for describing the drug.

        :return: The sources the model needs as input for describing the drug.
        """
        return [""]

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """Build the model.

        :param hyperparameters: The hyperparameters.
        """
        # Log hyperparameters to wandb if enabled
        self.log_hyperparameters(hyperparameters)

        self.hyperparameters = hyperparameters
        # init model in train

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
            raise ValueError("Drug input is required for XGDP")
        
        #step1 load data

        #step 2: preprocess data
        #to do 
        train_data = None
        test_data = None
        val_data = None

        #step 4: init model + train parameters
            #to do: get feature size from data? -> 
        with_attention = []
        model_name = self.hyperparameters["model"]
        model_class = getattr(models,model_name )
        self.model = model_class().to(device)
        
        if 'GAT' in model_name:
            return_attention_weights = True
        else:
            return_attention_weights = False
        n_epochs = self.hyperparameters["n_epochs"]
        lr = self.hyperparameters["lr"]
        train_batch_size = self.hyperparameters["train_batch_size"]
        test_batch_size = self.hyperparameters["test_batch_size"]
        val_batch_size = self.hyperparameters["val_batch_size"]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #step 3: build data loaders + split data
        test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
        train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False)
        
        #step 5: train loop
        self.model.train() #set model to train
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        #metrics for checking how training is going
        best_mse = 1000
        best_pearson = 0
        best_epoch = -1
        total_time = 0
        early_stop_tolerance = 30
        train_losses = []
        val_losses = []
        val_pearsons = []
        best_ret = []
        
        for epoch in n_epochs:
            #print what epoch is trained
            start_time = time.time()
            print(f"epoch : {epoch+1}/{n_epochs} ")
            #
            avg_loss = []
            for data in tqdm(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                # output, _ = model(data)
                x, x_cell_mut, edge_index, batch_drug, edge_feat = data.x, data.target, data.edge_index.long(), data.batch, data.edge_features
                # output, _ = model(x, edge_index, x_cell_mut, batch_drug, edge_feat)
                output = self.model(x, edge_index, batch_drug, x_cell_mut, edge_feat)
                loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
                loss.backward()
                optimizer.step()
                avg_loss.append(loss.item())
            train_loss = sum(avg_loss)/len(avg_loss)
        #step 6: save mdoel (maybe not neccecary beacuse class)



        
