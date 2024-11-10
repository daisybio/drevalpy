import torch.optim as optim
from torch.utils.data import DataLoader
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel
from typing import Any, Dict, Optional
from numpy.typing import ArrayLike
import torch
from torch import nn
import numpy as np


from .model_utils import Predictor
from .data_utils import (
    get_data,
    GraphDataset,
    CollateFn,
    load_expression_and_network_features,
    load_drug_feature_from_MolGNet,
)


class DIPK_Model(DRPModel):
    """
    DIPK model. Adapted from https://github.com/user15632/DIPK

    Improving drug response prediction via integrating gene relationships with deep learning
    Pengyong Li, Zhengxiang Jiang, Tianxiao Liu, Xinyu Liu, Hui Qiao, Xiaojun Yao
    Briefings in Bioinformatics, Volume 25, Issue 3, May 2024, bbae153, https://doi.org/10.1093/bib/bbae153

    """

    model_name = "DIPK"
    cell_line_views = ["gene_expression_features", "biological_network_features"]
    drug_views = ["drug_feature_embedding"]

    def build_model(self, hyperparameters: Dict[str, Any], *args, **kwargs):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Predictor(
            hyperparameters["embedding_dim"],
            hyperparameters["heads"],
            hyperparameters["fc_layer_num"],
            hyperparameters["fc_layer_dim"],
            hyperparameters["dropout_rate"],
        ).to(self.DEVICE)
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
        params = [{"params": self.model.parameters()}]
        optimizer = optim.Adam(params, lr=self.lr)

        # load data
        collate = CollateFn(train=True)
        graphs_train = get_data(
            cell_id=output.cell_line_ids,
            drug_id=output.drug_ids,
            cell_line_features=cell_line_input,
            drug_features=drug_input,
            ic50=output.response,
        )
        train_loader = DataLoader(
            GraphDataset(graphs_train), batch_size=self.batch_size, shuffle=True, collate_fn=collate
        )

        # train model
        for epoch in range(self.EPOCHS):
            self.model.train()
            epoch_loss = 0
            for it, (pyg_batch, gene_features, bionic_features) in enumerate(train_loader):
                pyg_batch, gene_features, bionic_features = (
                    pyg_batch.to(self.DEVICE),
                    gene_features.to(self.DEVICE),
                    bionic_features.to(self.DEVICE),
                )
                prediction = self.model(pyg_batch.x, pyg_batch, gene_features, bionic_features)
                loss = loss_func(torch.squeeze(prediction), pyg_batch.ic50)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            epoch_loss /= it + 1

    def predict(
        self,
        drug_ids: ArrayLike,
        cell_line_ids: ArrayLike,
        drug_input: FeatureDataset = None,
        cell_line_input: FeatureDataset = None,
    ) -> np.ndarray:

        # load data
        collate = CollateFn(train=False)
        Gtest = get_data(
            cell_id=cell_line_ids, drug_id=drug_ids, cell_line_features=cell_line_input, drug_features=drug_input
        )
        test_loader = DataLoader(GraphDataset(Gtest), batch_size=self.batch_size, shuffle=False, collate_fn=collate)

        # run prediction
        self.model.eval()
        test_pre = []
        with torch.no_grad():
            for it, (pyg_batch, gene_features, bionic_features) in enumerate(test_loader):
                pyg_batch, gene_features, bionic_features = (
                    pyg_batch.to(self.DEVICE),
                    gene_features.to(self.DEVICE),
                    bionic_features.to(self.DEVICE),
                )
                prediction = self.model(pyg_batch.x, pyg_batch, gene_features, bionic_features)
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
