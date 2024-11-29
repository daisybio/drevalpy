"""
Contains the SingleDrugRandomForest class.

It is a RandomForest model that uses only gene expression dataset for drug response prediction and trains one model
per drug.
"""

import numpy as np

from ...datasets.dataset import DrugResponseDataset, FeatureDataset
from .sklearn_models import RandomForest


class SingleDrugRandomForest(RandomForest):
    """SingleDrugRandomForest class."""

    is_single_drug_model = True
    drug_views = []
    early_stopping = False

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SingleDrugRandomForest
        """
        return "SingleDrugRandomForest"

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
    ) -> None:
        """
        Trains the model; the number of features is the number of fingerprints.

        :param output: training dataset containing the response output
        :param cell_line_input: training dataset containing gene expression data
        :param drug_input: not needed
        :param output_earlystopping: not needed
        :raises ValueError: if drug_input is not None
        """
        if drug_input is not None:
            raise ValueError("SingleDrugRandomForest does not support drug_input!")

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
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the drug response for the given cell lines.

        :param cell_line_ids: cell line ids
        :param drug_ids: drug ids, not needed here
        :param cell_line_input: cell line input
        :param drug_input: drug input, not needed here
        :returns: predicted drug response
        :raises ValueError: if drug_input is not None
        """
        if drug_input is not None:
            raise ValueError("drug_input is not needed.")

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
