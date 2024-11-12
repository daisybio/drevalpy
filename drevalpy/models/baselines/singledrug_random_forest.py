"""
Contains the SingleDrugRandomForest class.

It is a RandomForest model that uses only gene expression dataset for drug response prediction and trains one model
per drug.
"""

from typing import Optional

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from ..drp_model import SingleDrugModel
from .sklearn_models import RandomForest


class SingleDrugRandomForest(SingleDrugModel, RandomForest):
    """SingleDrugRandomForest class."""

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
        Trains the model; the number of features is the number of fingerprints.

        :param output: training dataset containing the response output
        :param cell_line_input: training dataset containing gene expression data
        :param drug_input: not needed
        :param output_earlystopping: not needed
        :raises ValueError: if drug_input or output_earlystopping is not None
        """
        if drug_input is not None or output_earlystopping is not None:
            raise ValueError("SingleDrugRandomForest does not support drug_input or " "output_earlystopping!")
        if len(output) > 0:
            x = self.get_concatenated_features(
                cell_line_view="gene_expression",
                drug_view=None,
                cell_line_ids_output=output.cell_line_ids,
                drug_ids_output=output.drug_ids,
                cell_line_input=cell_line_input,
                drug_input=None,
            )
            self.model.fit(x, output.response)
        else:
            print("No training data provided, will predict NA.")
            self.model = None

    def predict(
        self,
        drug_ids: str | np.ndarray,
        cell_line_ids: str | np.ndarray,
        drug_input: Optional[FeatureDataset] = None,
        cell_line_input: FeatureDataset = None,
    ) -> np.ndarray:
        """
        Predicts the drug response for the given cell lines.

        :param drug_ids: drug ids, not needed here
        :param cell_line_ids: cell line ids
        :param drug_input: drug input, not needed here
        :param cell_line_input: cell line input
        :returns: predicted drug response
        """
        if self.model is None:
            print("No training data was available, predicting NA.")
            return np.array([np.nan] * len(cell_line_ids))
        x = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view=None,
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=None,
        )
        return self.model.predict(x)
