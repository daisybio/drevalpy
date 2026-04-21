"""Contains XGDP, a GNN and CNN based drug response prediction model.

Drug discovery and mechanism prediction with explainable graph neural networks.

Original authors: Wang, C., Kumar, G.A. & Rajapakse, J.C. (2025, 10.1038/s41598-024-83090-3)
Code adapted from their Github: https://github.com/SCSE-Biomedical-Computing-Group/XGDP/blob/main/utils_tcnn.py
"""

import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any

import models
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import tqdm
import utils as u
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Subset
from torch.optim import Adam
from torch.utils.data import Dataset as PytorchDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from ...datasets.dataset import DrugResponseDataset, FeatureDataset
from ..drp_model import DRPModel
from ..lightning_metrics_mixin import RegressionMetricsMixin
from ..utils import load_and_select_gene_features
from .utils import XGDPPredictor, predict_model, train_epoch


class XGDP(DRPModel, RegressionMetricsMixin):
    """XGDP model for ..."""

    cell_line_views = ["gene_expression"]
    drug_views = ["drug_graph"]

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
        model_name = hyperparameters.get("model_type", "GATNet")
        #init in train
        #self.model = XGDPPredictor(name_hyperparameter=hyperparameter["name_hyperparameter"])
        """
         # Log hyperparameters to wandb if enabled
        self.log_hyperparameters(hyperparameters)

        self.hyperparameters = hyperparameters
        # init model in train
        """

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

        # step1 load data
        train_loader = self._prepare_dataloader(output, cell_line_input, drug_input, is_train=True)

        # step 2: preprocess data

        # step 3: build data loaders + split data
        #to do create split indices
        test_data = Subset(data, test_index)
        train_data = Subset(data, train_index)
        val_data = Subset(data, val_index)
        test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
        train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False)

        # step 4: init model + train parameters
        # to do: get feature size from data? ->

        #get model + init
        with_attention = []
        model_name = self.hyperparameters["model"]
        model_class = getattr(models, model_name)
        self.model = model_class().to(device) #add feature size
        if "GAT" in model_name:
            self.return_attention_weights = True
        else:
            self.return_attention_weights = False

        #get model hyperparameters
        n_epochs = self.hyperparameters["n_epochs"]
        lr = self.hyperparameters["lr"]
        train_batch_size = self.hyperparameters["train_batch_size"]
        test_batch_size = self.hyperparameters["test_batch_size"]
        val_batch_size = self.hyperparameters["val_batch_size"]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        do_save = self.hyperparameters["do_save"]
        
        #performance 
        log_interval = 20
        best_mse = 1000
        best_pearson = 0
        best_epoch = -1
        total_time = 0
        early_stop_tolerance = 30
        train_losses = []
        val_losses = []
        val_pearsons = []
        best_ret = []

        
        # step 5: train loop
        for epoch in tqdm(range(n_epochs)):
            start_time = time.time()
            train_loss = u.train(self.model, device, train_loader, optimizer, epoch+1, log_interval)
            G,P = u.predicting(self.model, device, val_loader)
            ret = [u.rmse(G,P),u.mse(G,P),u.pearson(G,P),u.spearman(G,P),u.coeffi_determ(G,P)]

            train_losses.append(train_loss)
            val_losses.append(ret[1])
            val_pearsons.append(ret[2])

            if ret[1]<best_mse:
                #rework to / check which folder is best?
                #if (do_save): #if enabeled saves best state of the model when model is better than previous versions
                #    torch.save(self.model.state_dict(), model_folder + model_file_name)
                    
                best_epoch = epoch+1
                best_mse = ret[1]
                best_pearson = ret[2]
                best_ret = ret

            total_time += time.time() - start_time
            if (epoch - best_epoch) > early_stop_tolerance:
                print('early stop at epoch ', epoch)
                break
        # test with the model with best validation performance
        if self.return_attention_weights:
            G_test, P_test, attn_weights = u.predicting(self.model, device, test_loader, self.return_attention_weights)
        else:
            G_test, P_test = u.predicting(self.model, device, test_loader)
        #rework 
        #result_file_name = 'result_' + save_name + '_' + dataset + '_' +  '.csv'
        #ret_test = [rmse(G_test,P_test),mse(G_test,P_test),pearson(G_test,P_test),spearman(G_test,P_test),coeffi_determ(G_test,P_test)]
        #if do_save:
        #    best_model_file_name = 'model_' + save_name + '_' + dataset + '_best' + str(best_model_id) +  '.model'
        #    torch.save(best_model.state_dict(), model_folder + best_model_file_name)
        #    if return_attention_weights:
        #        np.save(br_fol + '/Saliency/AttnWeight/' + model_st + '.npy', attn_weights)
        # step 6: save mdoel (maybe not neccecary beacuse class)

def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the response for the given input.

        :param drug_ids: list of drug ids, also used for single drug models, there it is just an array containing the
            same drug id
        :param cell_line_ids: list of cell line ids
        :param cell_line_input: input associated with the cell line, required for all models
        :param drug_input: input associated with the drug, optional because single drug models do not use drug features
        :returns: predicted response
        """
        #step 1 load data
        data = None
        #(step 2 preprocess)

        #step 3 dataloaders + 
        loader = DataLoader(data, batch_size=1, shuffle=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #step 4 predict
        ret_att = self.return_attention_weights
        pred = u.predicting(model = self.model,device = device,loader = loader, return_attention_weights= ret_att)
        #step 5 convert to dreval datatype + return