"""
Contains the MOLIR model, a regression adaptation of the MOLI model.

Original authors: Sharifi-Noghabi et al. (2019, 10.1093/bioinformatics/btz318)
Code adapted from their Github: https://github.com/hosseinshn/MOLI
and Hauptmann et al. (2023, 10.1186/s12859-023-05166-7) https://github.com/kramerlab/Multi-Omics_analysis
"""

from typing import Any

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from ...datasets.dataset import DrugResponseDataset, FeatureDataset
from ..drp_model import DRPModel
from ..utils import get_multiomics_feature_dataset
from .utils import MOLIModel, get_dimensions_of_omics_data


class MOLIR(DRPModel):
    """
    Regression extension of MOLI: multi-omics late integration deep neural network.

    Takes somatic mutation, copy number variation and gene expression data as input. MOLI uses type-specific encoding
    subnetworks to learn features for each omics type, concatenates them into one representation and optimizes this
    representation via a combined cost function consisting of a triplet loss and a binary cross-entropy loss.
    We use a regression adaption with MSE loss and a mechanism to find positive and negative samples.
    """

    is_single_drug_model = True
    cell_line_views = ["gene_expression", "mutations", "copy_number_variation_gistic"]
    drug_views = []
    early_stopping = True

    def __init__(self) -> None:
        """
        Initializes the MOLIR model.

        The hyperparameters are set in build_model, the model is set in train when we know the dimensionality of the
        gene expression, mutation and copy number variation data.
        """
        super().__init__()
        self.model: MOLIModel | None = None
        self.hyperparameters: dict[str, Any] = dict()

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: MOLIR
        """
        return "MOLIR"

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Builds the model from hyperparameters.

        :param hyperparameters: Custom hyperparameters for the model, includes mini_batch, layer dimensions (h_dim1,
            h_dim2, h_dim3), learning_rate, dropout_rate, weight_decay, gamma, epochs, and margin.
        """
        self.hyperparameters = hyperparameters

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
    ) -> None:
        """
        Initializes and trains the model.

        First, the gene expression data is reduced using a variance threshold (0.05) and standardized. Then,
        the model is initialized with the hyperparameters and the dimensions of the gene expression, mutation and
        copy number variation data. If there is no training data, the model is set to None (and predictions will be
        skipped as well). If there is not enough training data, the predictions will be made on the randomly
        initialized model.

        :param output: drug response data
        :param cell_line_input: cell line omics features, i.e., gene expression, mutations and copy number variation
        :param drug_input: drug features, not needed
        :param output_earlystopping: early stopping data, not used when there is not enough data
        """
        if len(output) > 0:
            selector_gex = VarianceThreshold(0.05)
            cell_line_input.fit_transform_features(
                train_ids=np.unique(output.cell_line_ids),
                transformer=selector_gex,
                view="gene_expression",
            )
            scaler_gex = StandardScaler()
            cell_line_input.fit_transform_features(
                train_ids=np.unique(output.cell_line_ids),
                transformer=scaler_gex,
                view="gene_expression",
            )
            if output_earlystopping is not None and self.early_stopping and len(output_earlystopping) < 2:
                output_earlystopping = None
            dim_gex, dim_mut, dim_cnv = get_dimensions_of_omics_data(cell_line_input)
            self.model = MOLIModel(
                hpams=self.hyperparameters,
                input_dim_expr=dim_gex,
                input_dim_mut=dim_mut,
                input_dim_cnv=dim_cnv,
            )
            if len(output) >= self.hyperparameters["mini_batch"]:
                self.model.fit(
                    output_train=output,
                    cell_line_input=cell_line_input,
                    output_earlystopping=output_earlystopping,
                )
            else:
                print(f"Not enough training data provided ({len(output)}), will predict on randomly initialized model.")
        else:
            print("No training data provided, skipping model")
            self.model = None

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the drug response.

        If there was no training data, only nans will be returned.

        :param cell_line_ids: Cell lines to predict
        :param drug_ids: Drugs to predict
        :param cell_line_input: cell line omics features
        :param drug_input: drug features, not needed
        :returns: Predicted drug response
        """
        input_data = self.get_feature_matrices(
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        (gene_expression, mutations, cnvs) = (
            input_data["gene_expression"],
            input_data["mutations"],
            input_data["copy_number_variation_gistic"],
        )
        if self.model is None:
            print("No model trained, will predict NA.")
            return np.array([np.nan] * len(cell_line_ids))
        return self.model.predict(gene_expression, mutations, cnvs)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features: gene expression, mutations and copy number variation.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset with gene expression, mutations and copy number variation
        """
        feature_dataset = get_multiomics_feature_dataset(
            data_path=data_path,
            dataset_name=dataset_name,
            gene_list=None,
            omics=self.cell_line_views,
        )
        # log transformation
        feature_dataset.apply(function=np.log, view="gene_expression")
        return feature_dataset

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """
        Returns None, as drug features are not needed for MOLIR.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: None
        """
        return None
