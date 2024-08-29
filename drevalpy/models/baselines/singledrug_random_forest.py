"""
Contains the SingleDrugRandomForest class, which is a RandomForest model that uses only gene
expression dataset for drug response prediction and trains one model per drug.
"""

from typing import Optional
import numpy as np
from numpy.typing import ArrayLike

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from ..drp_model import SingleDrugModel
from .sklearn_models import RandomForest


class SingleDrugRandomForest(SingleDrugModel, RandomForest):
    """
    SingleDrugRandomForest class.
    """

    drug_views = []
    model_name = "SingleDrugRandomForest"
    early_stopping = False

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input=None,
        output_earlystopping=None,
    ) -> None:
        """
        Trains the model: the number of features is the number of fingerprints.
        :param **kwargs:
        :param output: training dataset containing the response output
        :param cell_line_input: training dataset containing gene expression data
        :param drug_input: not needed
        :param output_earlystopping: not needed
        """
        if drug_input is not None or output_earlystopping is not None:
            raise ValueError(
                "SingleDrugRandomForest does not support drug_input or "
                "output_earlystopping!"
            )

        x = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view=None,
            cell_line_ids_output=output.cell_line_ids,
            drug_ids_output=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=None,
        )
        self.model.fit(x, output.response)

    def predict(
        self,
        drug_ids: ArrayLike,
        cell_line_ids: ArrayLike,
        drug_input: Optional[FeatureDataset] = None,
        cell_line_input: FeatureDataset = None,
    ) -> np.ndarray:
        x = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view=None,
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=None,
        )
        return self.model.predict(x)
