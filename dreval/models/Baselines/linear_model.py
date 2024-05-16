from typing import List
import numpy as np
from sklearn.linear_model import ElasticNet, Ridge, LogisticRegression

from dreval.dataset import FeatureDataset, DrugResponseDataset
from dreval.models.drp_model import DRPModel
from ..utils import load_ge_features_from_landmark_genes, load_drug_features_from_fingerprints


class ElasticNetModel(DRPModel):
    model_name = 'ElasticNet'
    cell_line_views = ['gene_expression']
    drug_views = ['fingerprints']

    @staticmethod
    def get_hyperparameter_set() -> List[dict]:
        """
        Returns a list of hyperparameters for the model.
        Hyperparameters to consider:
        - alpha: Constant that multiplies the penalty terms.
        - l1_ratio: The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
        - feature_path: Path to the feature dataset.
        :return: List of hyperparameters including the default combination, lasso and ridge.
        """
        hpams = [
            # elastic net default hyperparameters
            {'alpha': 1.0, 'l1_ratio': 0.5},
            # only l1 regularization = lasso
            {'alpha': 1.0, 'l1_ratio': 1.0},
            # only l2 regularization = ridge
            {'alpha': 1.0, 'l1_ratio': 0.0}
        ]
        for hpam in hpams:
            hpam['feature_path'] = 'data/GDSC'
        return hpams

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.
        :param hyperparameters: Hyperparameters for the model.
        """
        if hyperparameters['l1_ratio'] == 0.0:
            self.model = Ridge(alpha=hyperparameters['alpha'])
        else:
            self.model = ElasticNet(alpha=hyperparameters['alpha'],
                                    l1_ratio=hyperparameters['l1_ratio'])

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

    def predict(self,
                gene_expression: np.ndarray = None,
                fingerprints: np.ndarray = None) -> np.ndarray:
        """
        Predicts the drug response.
        :param gene_expression:
        :param fingerprints:
        :return: predicted response
        """
        X = np.concatenate((gene_expression, fingerprints), axis=1)
        return self.model.predict(X)

    def save(self, path):
        raise NotImplementedError('ElasticNetModel does not support saving yet ...')

    def load(self, path):
        raise NotImplementedError('ElasticNetModel does not support loading yet ...')

    def load_cell_line_features(self, path: str) -> FeatureDataset:
        """
        Loads the cell line features.
        :param path: Path to the gene expression and landmark genes
        :return: FeatureDataset containing the cell line gene expression features, filtered through the landmark genes
        """
        return load_ge_features_from_landmark_genes(path)

    def load_drug_features(self, path: str) -> FeatureDataset:
        return load_drug_features_from_fingerprints(path)
