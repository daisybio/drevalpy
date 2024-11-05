import torch.optim as optim
from torch.utils.data import DataLoader
import time
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel
from typing import Any, Dict, Optional
from numpy.typing import ArrayLike

import pandas as pd
import numpy as np
import os
import joblib

from .Model import *
from .Data import *

class DIPK_Model(DRPModel):
    
    model_name = "DIPK"
    cell_line_views = ["gene_expression_features", "biological_network_features"]
    drug_views = ["drug_feature_embedding"]

    def build_model(self, hyperparameters: Dict[str, Any], *args, **kwargs):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Predictor(hyperparameters["embedding_dim"], hyperparameters["heads"], hyperparameters["fc_layer_num"], hyperparameters["fc_layer_dim"], hyperparameters["dropout_rate"]).to(self.DEVICE)
        self.EPOCHS = hyperparameters["EPOCHS"]
        self.batch_size = hyperparameters["batch_size"]
        self.lr = hyperparameters["lr"]
   
    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: Optional[FeatureDataset] = None,
        output_earlystopping: Optional[DrugResponseDataset] = None,
    ) -> None:
        
        loss_func = nn.MSELoss()
        params = [
            {'params': self.model.parameters()}
        ]
        optimizer = optim.Adam(params, lr=self.lr)
        
        # load data
        my_collate = CollateFn_Train()
        Gtrain = GetTrainData(output.cell_line_ids, output.drug_ids, output.response, cell_line_input, drug_input)
        train_loader = DataLoader(MyDataSet(Gtrain), batch_size=self.batch_size, shuffle=True, collate_fn=my_collate)
        
        # train model
        for epoch in range(self.EPOCHS):
            self.model.train()
            epoch_loss = 0
            for it, (pyg_batch, GeneFt, BionicFt) in enumerate(train_loader):
                pyg_batch, GeneFt, BionicFt = pyg_batch.to(self.DEVICE), GeneFt.to(self.DEVICE), BionicFt.to(self.DEVICE)
                prediction = self.model(pyg_batch.x, pyg_batch, GeneFt, BionicFt)
                loss = loss_func(torch.squeeze(prediction), pyg_batch.ic50)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            epoch_loss /= (it + 1)
              
    def predict(
        self,
        drug_ids: ArrayLike,
        cell_line_ids: ArrayLike,
        drug_input: FeatureDataset = None,
        cell_line_input: FeatureDataset = None,
    ) -> np.ndarray:
        
        #load data
        my_collate = CollateFn_Test()
        Gtest = GetTestData(cell_line_ids, drug_ids, cell_line_input, drug_input)
        test_loader = DataLoader(MyDataSet(Gtest), batch_size=self.batch_size, shuffle=False, collate_fn=my_collate)
        
        #run prediction
        self.model.eval()
        test_pre = []
        with torch.no_grad():
            for it, (pyg_batch, GeneFt, BionicFt) in enumerate(test_loader):
                pyg_batch, GeneFt, BionicFt = pyg_batch.to(self.DEVICE), GeneFt.to(self.DEVICE), BionicFt.to(self.DEVICE)
                prediction = self.model(pyg_batch.x, pyg_batch, GeneFt, BionicFt)
                test_pre += torch.squeeze(prediction).cpu().tolist()
               
        return test_pre
        
    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
         
        return load_expression_and_network_features(
            feature_type1=self.cell_line_views[0],
            feature_type2=self.cell_line_views[1],
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:

        return load_drug_feature_from_MolGNet(
            feature_type=self.drug_views[0],
            feature_subtype1="MolGNet_features",
            feature_subtype2="Edge_Index",
            feature_subtype3="Edge_Attr",
            data_path=data_path, 
            dataset_name=dataset_name,
            )