from typing import Optional
from models.SimpleNeuralNetwork.utils import FeedForwardNetwork
from dreval.drp_model import DRPModel
from dreval.datasets.dataset import DrugResponseDataset, FeatureDataset
import numpy as np
import pandas as pd
import warnings


class SimpleNeuralNetwork(DRPModel):
    """
    Simple Feedforward Neural Network model with dropout.
    hyperparameters:
        units_per_layer: number of units per layer e.g. [100, 50] means 2 layers with 100 and 50 units respectively and the output layer with one unit.
        dropout_prob: dropout probability for layers 1, 2, ..., n-1
    """

    cell_line_views = ["gene_expression"]

    drug_views = ["fingerprints"]

    early_stopping = True

    model_name = "SimpleNeuralNetwork"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.
        """
        self.model = FeedForwardNetwork(
            n_features=hyperparameters["n_features"],
            n_units_per_layer=hyperparameters["units_per_layer"],
            dropout_prob=hyperparameters["dropout_prob"],
        )

    def train(
        self,
        output: DrugResponseDataset,
        output_earlystopping: Optional[DrugResponseDataset] = None,
        gene_expression: np.ndarray = None,
        fingerprints: np.ndarray = None,
        gene_expression_earlystopping: Optional[np.ndarray] = None,
        fingerprints_earlystopping: Optional[np.ndarray] = None,
    ):
        """
        Trains the model.
        :param output: training data associated with the response output
        :param output_earlystopping: optional early stopping dataset
        :param gene_expression: gene expression data
        :param fingerprints: fingerprints data
        :param gene_expression_earlystopping: gene expression data for early stopping
        :param fingerprints_earlystopping: fingerprints data for early stopping

        """
        X = np.concatenate((gene_expression, fingerprints), axis=1)

        if all([ar is not None for ar in [output_earlystopping, gene_expression_earlystopping, fingerprints_earlystopping]]):
            X_earlystopping = np.concatenate(
                (gene_expression_earlystopping, fingerprints_earlystopping), axis=1
            )
        else:
            X_earlystopping = None
        
        if output_earlystopping is not None:
            response_earlystopping = output_earlystopping.response
        else:
            response_earlystopping = None

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*does not have many workers which may be a bottleneck.*",
            )
            self.model.fit(
                X_train=X,
                y_train=output.response,
                X_eval=X_earlystopping,
                y_eval=response_earlystopping,
                batch_size=16,
                patience=5,
                num_workers=1,
            )

    def save(self, path: str):
        """
        Saves the model.
        :param path: path to save the model
        """
        self.model.save(path)

    @staticmethod
    def load(path: str):
        # TODO
        raise NotImplementedError("load method not implemented")

    def predict(
        self,
        gene_expression: np.ndarray = None,
        fingerprints: np.ndarray = None
    ) -> np.ndarray:
        """
        Predicts the response for the given input.
        """
        X = np.concatenate((gene_expression, fingerprints), axis=1)
        return self.model.predict(X)

    def load_cell_line_features(self, path: str) -> FeatureDataset:
        """
        Fetch cell line input data
        :return: FeatureDataset
        """
        ge = pd.read_csv(f"{path}/gene_expression.csv", index_col=0)
        landmark_genes = pd.read_csv(f"{path}/gene_lists/landmark_genes.csv", sep="\t")
        genes_to_use = set(landmark_genes["Symbol"]) & set(ge.columns)
        ge = ge[list(genes_to_use)]

        return FeatureDataset(
            {cl: {"gene_expression": ge.loc[cl].values} for cl in ge.index}
        )

    def load_drug_features(self, path: str) -> FeatureDataset:
        """
        Fetch drug input data.
        :return: FeatureDataset
        """
        fingerprints = pd.read_csv(
            f"{path}/drug_fingerprints/drug_name_to_demorgan_128_map.csv", index_col=0
        ).T
        return FeatureDataset(
            {
                drug: {"fingerprints": fingerprints.loc[drug].values}
                for drug in fingerprints.index
            }
        )

    def get_hyperparameter_set():
        hpams = [
            {"dropout_prob": 0.2, "units_per_layer": [10, 10, 10]},
            {"dropout_prob": 0.3, "units_per_layer": [20, 10, 10]},
        ]
        for hpam in hpams:
            hpam["feature_path"] = "data/GDSC"
            hpam["n_features"] = 1036
        return hpams
