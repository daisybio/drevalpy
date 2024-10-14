"""
Code for the MOLI model.
Original authors: Sharifi-Noghabi et al. (2019, 10.1093/bioinformatics/btz318)
Code adapted from: Hauptmann et al. (2023, 10.1186/s12859-023-05166-7), https://github.com/kramerlab/Multi-Omics_analysis
"""

import subprocess
from io import BytesIO
import pandas as pd
from typing import Optional, List
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from ..utils import RegressionDataset


class MOLIEncoder(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(MOLIEncoder, self).__init__()
        self.encode = torch.nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.encode(x)


class MOLIClassifier(nn.Module):
    def __init__(self, input_size, dropout_rate):
        super(MOLIClassifier, self).__init__()
        self.classifier = torch.nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, 1),
        )

    def forward(self, x):
        return self.classifier(x)


class Moli(nn.Module):
    def __init__(self, hpams):
        super().__init__()
        self.mini_batch = hpams["mini_batch"]
        self.h_dim1 = hpams["h_dim1"]
        self.h_dim2 = hpams["h_dim2"]
        self.h_dim3 = hpams["h_dim3"]
        self.lr_e = hpams["lr_e"]
        self.lr_m = hpams["lr_m"]
        self.lr_c = hpams["lr_c"]
        self.lr_clf = hpams["lr_cl"]
        self.dropout_rate_e = hpams["dropout_rate_e"]
        self.dropout_rate_m = hpams["dropout_rate_m"]
        self.dropout_rate_c = hpams["dropout_rate_c"]
        self.dropout_rate_clf = hpams["dropout_rate_clf"]
        self.weight_decay = hpams["weight_decay"]
        self.gamma = hpams["gamma"]
        self.epochs = hpams["epochs"]
        self.margin = hpams["margin"]
        # not BCE or triplet loss because we're treating it as regression problem
        # TODO Triplet Regression Loss
        self.loss = nn.MSELoss()
        self.model_initialized = False
        self.expression_encoder = None
        self.mutation_encoder = None
        self.cna_encoder = None
        self.classifier = None

    def initialize_model(self, x_train_e, x_train_m, x_train_c):
        _, ie_dim = x_train_e.shape
        _, im_dim = x_train_m.shape
        _, ic_dim = x_train_c.shape
        self.expression_encoder = MOLIEncoder(ie_dim, self.h_dim1, self.dropout_rate_e)
        self.mutation_encoder = MOLIEncoder(im_dim, self.h_dim2, self.dropout_rate_m)
        self.cna_encoder = MOLIEncoder(ic_dim, self.h_dim3, self.dropout_rate_c)
        self.classifier = MOLIClassifier(
            ie_dim + im_dim + ic_dim, self.dropout_rate_clf
        )
        self.model_initialized = True

    def fit(
        self,
        output_train: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset = None,
        cell_line_views: List[str] = None,
        drug_views: List[str] = None,
        output_earlystopping: Optional[DrugResponseDataset] = None,
        patience: int = 5,
    ):
        device = create_device(gpu_number=None)
        train_dataset = RegressionDataset(
            output=output_train,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            cell_line_views=cell_line_views,
            drug_views=drug_views,
        )
        # no weighted random sampler because we are not treating it as a classification problem
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.mini_batch,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )
        val_loader = None
        if output_earlystopping is not None:
            val_dataset = RegressionDataset(
                output=output_earlystopping,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
                cell_line_views=cell_line_views,
                drug_views=drug_views,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.mini_batch,
                shuffle=False,
                num_workers=1,
                persistent_workers=True,
            )
        moli_optimiser = torch.optim.Adagrad(
            [
                {"params": self.expression_encoder.parameters(), "lr": self.lr_e},
                {"params": self.mutation_encoder.parameters(), "lr": self.lr_m},
                {"params": self.cna_encoder.parameters(), "lr": self.lr_c},
                {"params": self.classifier.parameters(), "lr": self.lr_clf},
            ],
            weight_decay=self.weight_decay,
        )
        cpu = device == torch.device("cpu")

        for _ in range(self.epochs):
            for batch in train_loader:
                x_train_e, x_train_m, x_train_c, y_train = batch
                x_train_e = x_train_e.to(device)
                x_train_m = x_train_m.to(device)
                x_train_c = x_train_c.to(device)
                y_train = y_train.to(device)
                self.expression_encoder = self.expression_encoder.to(device)
                self.expression_encoder.train()
                self.mutation_encoder = self.mutation_encoder.to(device)
                self.mutation_encoder.train()
                self.cna_encoder = self.cna_encoder.to(device)
                self.cna_encoder.train()
                self.classifier = self.classifier.to(device)
                self.classifier.train()

                z_ex = self.expression_encoder(x_train_e)
                z_mu = self.mutation_encoder(x_train_m)
                z_cn = self.cna_encoder(x_train_c)

                z = torch.cat((z_ex, z_mu, z_cn), 1)
                z = F.normalize(z, p=2, dim=0)
                preds = self.classifier(z)
                loss = self.loss(preds, y_train)
                moli_optimiser.zero_grad()
                loss.backward()
                moli_optimiser.step()
                # early stopping
                if val_loader is not None:
                    with torch.no_grad():
                        self.expression_encoder.eval()
                        self.mutation_encoder.eval()
                        self.cna_encoder.eval()
                        self.classifier.eval()


    def forward_with_features(self, expression, mutation, cna):
        left_out = self.expression_encoder(expression)
        middle_out = self.mutation_encoder(mutation)
        right_out = self.cna_encoder(cna)
        left_middle_right = torch.cat((left_out, middle_out, right_out), 1)
        return [self.classifier(left_middle_right), left_middle_right]

    def forward(self, expression, mutation, cna):
        if not self.model_initialized:
            self.initialize_model(expression, mutation, cna)
        left_out = self.expression_encoder(expression)
        middle_out = self.mutation_encoder(mutation)
        right_out = self.cna_encoder(cna)
        left_middle_right = torch.cat((left_out, middle_out, right_out), 1)
        return self.classifier(left_middle_right)


def create_device(gpu_number):
    if torch.cuda.is_available():
        if gpu_number is None:
            free_gpu_id = get_free_gpu()
        else:
            free_gpu_id = gpu_number
        device = torch.device(f"cuda:{free_gpu_id}")
    else:
        device = torch.device("cpu")
    return device


def get_free_gpu():
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.free"]
    )
    gpu_df = pd.read_csv(BytesIO(gpu_stats), names=["memory.free"], skiprows=1)
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
    return gpu_df["memory.free"].idxmax()



