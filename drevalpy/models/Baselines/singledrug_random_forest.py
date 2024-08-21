from typing import Optional
import numpy as np
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import SingleDrugModel
from .random_forest import RandomForest
from numpy.typing import ArrayLike


class SingleDrugRandomForest(SingleDrugModel, RandomForest):
    drug_views = []
    model_name = "SingleDrugRandomForest"
    early_stopping = False

    def train(
        self, output: DrugResponseDataset, cell_line_input: FeatureDataset, **kwargs
    ) -> None:
        """
        Trains the model: the number of features is the number of fingerprints.
        :param **kwargs:
        :param output: training dataset containing the response output
        :param gene_expression: training gene expression data
        """
        X = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view=None,
            cell_line_ids_output=output.cell_line_ids,
            drug_ids_output=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=None,
        )
        self.model.fit(X, output.response)

    def predict(
        self,
        drug_ids: ArrayLike,
        cell_line_ids: ArrayLike,
        drug_input: Optional[FeatureDataset] = None,
        cell_line_input: FeatureDataset = None,
        **kwargs
    ) -> np.ndarray:
        X = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view=None,
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=None,
        )
        return self.model.predict(X)
