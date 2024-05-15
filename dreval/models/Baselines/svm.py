from typing import List
import numpy as np
from sklearn.svm import SVR

from dreval.dataset import FeatureDataset, DrugResponseDataset
from dreval.drp_model import DRPModel
from ..utils import load_ge_features_from_landmark_genes, load_drug_features_from_fingerprints


class SVMRegressor(DRPModel):
    model_name = 'SVR'
    cell_line_views = ['gene_expression']
    drug_views = ['fingerprints']

    @staticmethod
    def get_hyperparameter_set() -> List[dict]:
        """
        Returns a list of hyperparameters for the model.
        Hyperparameters to consider:
        - C: Regularization parameter. The strength of the regularization is inversely proportional to C.
        - epsilon: Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.
        - kernel: Specifies the kernel type to be used in the algorithm.
        - feature_path: Path to the feature dataset.
        :return: List of hyperparameters including the default combination.
        """
        hpams = [
            {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1},
            {'kernel': 'linear', 'C': 1.0, 'epsilon': 0.1},
            {'kernel': 'poly', 'C': 1.0, 'epsilon': 0.1},
            {'kernel': 'sigmoid', 'C': 1.0, 'epsilon': 0.1}
        ]
        for hpam in hpams:
            hpam['feature_path'] = 'data/GDSC'
        return hpams

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.
        :param hyperparameters: Hyperparameters for the model.
        """
        self.model = SVR(kernel=hyperparameters['kernel'],
                         C=hyperparameters['C'],
                         epsilon=hyperparameters['epsilon'])

    def train(self, output: DrugResponseDataset,
              gene_expression: np.ndarray = None,
              fingerprints: np.ndarray = None) -> None:
        """
        Trains the model: the number of features is the number of genes + the number of fingerprints.
        :param output: training dataset containing the response output
        :param gene_expression: training dataset containing gene expression data
        :param fingerprints: training dataset containing fingerprints data
        """
        X = np.concatenate((gene_expression, fingerprints), axis=1)
        self.model.fit(X, output.response)

    def predict(self, gene_expression: np.ndarray = None,
                fingerprints: np.ndarray = None) -> np.ndarray:
        """
        Predicts the response values.
        :param gene_expression:
        :param fingerprints:
        :return:
        """
        X = np.concatenate((gene_expression, fingerprints), axis=1)
        return self.model.predict(X)

    def save(self, path):
        raise NotImplementedError('SVR does not support saving yet ...')

    def load(self, path):
        raise NotImplementedError('SVR does not support loading yet ...')

    def load_cell_line_features(self, path: str) -> FeatureDataset:
        """
        Loads the cell line features.
        :param path: Path to the gene expression and landmark genes
        :return: FeatureDataset containing the cell line gene expression features, filtered through the landmark genes
        """
        return load_ge_features_from_landmark_genes(path)

    def load_drug_features(self, path: str) -> FeatureDataset:
        return load_drug_features_from_fingerprints(path)