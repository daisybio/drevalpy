"""
DIPK model. Adapted from https://github.com/user15632/DIPK.

Original publication:
Improving drug response prediction via integrating gene relationships with deep learning
Pengyong Li, Zhengxiang Jiang, Tianxiao Liu, Xinyu Liu, Hui Qiao, Xiaojun Yao
Briefings in Bioinformatics, Volume 25, Issue 3, May 2024, bbae153, https://doi.org/10.1093/bib/bbae153
"""

from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.utils import load_and_reduce_gene_features

from .data_utils import CollateFn, GraphDataset, get_data, load_bionic_features, load_drug_feature_from_mol_g_net
from .gene_expression_encoder import GeneExpressionEncoder, encode_gene_expression, train_gene_expession_autoencoder
from .model_utils import Predictor


class DIPKModel(DRPModel):
    """DIPK model. Adapted from https://github.com/user15632/DIPK."""

    cell_line_views = ["gene_expression", "bionic_features"]
    drug_views = ["drug_feature_embedding"]

    def __init__(self) -> None:
        """Initialize the DIPK model."""
        super().__init__()
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # all of this gets initialized in build_model
        self.model: Predictor | None = None
        self.EPOCHS: int = 0
        self.batch_size: int = 0
        self.lr: float = 0.0
        self.gene_expression_encoder: GeneExpressionEncoder | None = None

    @classmethod
    def get_model_name(cls) -> str:
        """
        Get the model name.

        :returns: DIPK
        """
        return "DIPK"

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Builds the DIPK model with the specified hyperparameters.

        :param hyperparameters: embedding_dim, heads, fc_layer_num, fc_layer_dim, dropout_rate, EPOCHS, batch_size, lr

        Details of hyperparameters:

        - embedding_dim: int, embedding dimension used for the graph encoder which is not used in the final model
        - heads: int, number of heads for the multi-head attention layer, defaults to 1
        - fc_layer_num: int, number of fully connected layers for the dense layers
        - fc_layer_dim: list[int], number of neurons for each fully connected layer
        - dropout_rate: float, dropout rate for all fully connected layers
        - EPOCHS: int, number of epochs to train the model
        - batch_size: int, batch size for training
        - lr: float, learning rate for training
        """
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
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
    ) -> None:
        """
        Trains the model.

        :param output: training data associated with the response output
        :param cell_line_input: input data associated with the cell line
        :param drug_input: input data associated with the drug
        :param output_earlystopping: early stopping data associated with the response output, not used
        :raises ValueError: if drug_input is None or if the model is not initialized
        """
        if drug_input is None:
            raise ValueError("DIPK model requires drug features.")
        if not isinstance(self.model, Predictor):
            raise ValueError("DIPK model not initialized.")

        loss_func = nn.MSELoss()
        params = [{"params": self.model.parameters()}]
        optimizer = optim.Adam(params, lr=self.lr)

        self.gene_expression_encoder = train_gene_expession_autoencoder(
            cell_line_input.get_feature_matrix(view="gene_expression", identifiers=output.cell_line_ids)
        )

        cell_line_input.apply(lambda x: encode_gene_expression(x, self.gene_expression_encoder), view="gene_expression")

        # load data
        collate = CollateFn(train=True)
        graphs_train = get_data(
            cell_ids=output.cell_line_ids,
            drug_ids=output.drug_ids,
            cell_line_features=cell_line_input,
            drug_features=drug_input,
            ic50=output.response,
        )

        train_loader: DataLoader = DataLoader(
            GraphDataset(graphs_train), batch_size=self.batch_size, shuffle=True, collate_fn=collate
        )

        # train model
        for _ in range(self.EPOCHS):
            self.model.train()
            for pyg_batch, gene_features, bionic_features in train_loader:
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

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the response values for the given cell lines and drugs.

        :param cell_line_ids: list of cell line IDs
        :param drug_ids: list of drug IDs
        :param cell_line_input: input data associated with the cell line
        :param drug_input: input data associated with the drug
        :return: predicted response values
        :raises ValueError: if drug_input is None or if the model is not initialized
        """
        if drug_input is None:
            raise ValueError("DIPK model requires drug features.")
        if not isinstance(self.model, Predictor):
            raise ValueError("DIPK model not initialized.")

        # load data
        collate = CollateFn(train=False)
        gtest = get_data(
            cell_ids=cell_line_ids, drug_ids=drug_ids, cell_line_features=cell_line_input, drug_features=drug_input
        )
        test_loader: DataLoader = DataLoader(
            GraphDataset(gtest), batch_size=self.batch_size, shuffle=False, collate_fn=collate
        )

        # run prediction
        self.model.eval()
        test_pre = []
        with torch.no_grad():
            for _, (pyg_batch, gene_features, bionic_features) in enumerate(test_loader):
                pyg_batch, gene_features, bionic_features = (
                    pyg_batch.to(self.DEVICE),
                    gene_features.to(self.DEVICE),
                    bionic_features.to(self.DEVICE),
                )
                prediction = self.model(pyg_batch.x, pyg_batch, gene_features, bionic_features)
                test_pre += torch.squeeze(prediction).cpu().tolist()

        return np.array(test_pre)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load cell line features.

        :param data_path: path to the data
        :param dataset_name: path to the dataset
        :returns: cell line features
        """
        gene_expression = load_and_reduce_gene_features(
            feature_type="gene_expression",
            gene_list=None,
            data_path=data_path,
            dataset_name=dataset_name,
        )
        bionic_features = load_bionic_features(
            data_path=data_path,
            dataset_name=dataset_name,
        )
        return bionic_features.add_features(gene_expression)

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load drug features.

        :param data_path: path to the data
        :param dataset_name: path to the dataset
        :returns: drug features
        """
        return load_drug_feature_from_mol_g_net(
            feature_type=self.drug_views[0],
            feature_subtype1="MolGNet_features",
            feature_subtype2="Edge_Index",
            feature_subtype3="Edge_Attr",
            data_path=data_path,
            dataset_name=dataset_name,
        )
