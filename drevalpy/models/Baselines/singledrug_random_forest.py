import numpy as np
from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models.drp_model import SingleDrugModel
from .random_forest import RandomForest


class SingleDrugRandomForest(SingleDrugModel, RandomForest):
    drug_views = []
    model_name = "SingleDrugRandomForest"
    early_stopping = False

    def train(
        self, output: DrugResponseDataset, gene_expression: np.ndarray = None
    ) -> None:
        """
        Trains the model: the number of features is the number of fingerprints.
        :param output: training dataset containing the response output
        :param gene_expression: training gene expression data
        """
        self.model.fit(gene_expression, output.response)

    def predict(self, gene_expression: np.ndarray = None) -> np.ndarray:
        """
        Predicts the response for the given input.
        :param gene_expression: gene expression data
        :return: predicted response
        """
        return self.model.predict(gene_expression)
