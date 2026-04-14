import pandas as pd
import numpy as np
import sys 
import os
import torch
import torch.nn as nn
import tqdm
import time

import models
import utils as u

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
        
        #step 5: train loop
        best_model_id = 0
        best_model = None
        best_pearson_cv = 0
        ret_cv = []
        kf = KFold(n_splits=3) #creates splits indices for cross validation
        #does the cv split
        for i, (train_index, val_index) in enumerate(kf.split(cv_data)):
            #splits data into val and train according to index form kf
            train_data = Subset(data,train_index)
            train_data = Subset(data,val_index)
            train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=False)
            val_loader = DataLoader(test_data, batch_size=val_batch_size, shuffle=False)

            #creates model for each cv split
            model = model_class().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            best_mse = 1000
            best_pearson = 0
            best_epoch = -1
            total_time = 0
            early_stop_tolerance = 30
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



        
