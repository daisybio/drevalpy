"""
Contains the MOLIR model.
Original authors: Sharifi-Noghabi et al. (2019, 10.1093/bioinformatics/btz318)
Code adapted from their Github: https://github.com/hosseinshn/MOLI
and Hauptmann et al. (2023, 10.1186/s12859-023-05166-7) https://github.com/kramerlab/Multi-Omics_analysis
"""

from typing import Any, Optional

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from ...datasets.dataset import DrugResponseDataset, FeatureDataset
from ..drp_model import SingleDrugModel
from ..utils import load_and_reduce_gene_features
from .utils import MOLIModel, get_dimensions_of_omics_data


class MOLIR(SingleDrugModel):
    """
    Regression extension of
    MOLI: multi-omics late integration deep neural network.
    Takes somatic mutation, copy number variation and gene expression data as input.
    MOLI uses type-specific encoding subnetworks to learn features for each omics type,
    concatenates them into one representation and optimizes this representation via a combined cost
    function consisting of a triplet loss and a binary cross-entropy loss.
    We use a regression adaption with MSE loss and an mechanism to find positive and negative samples.
    """

    cell_line_views = ["gene_expression", "mutations", "copy_number_variation_gistic"]
    drug_views = []
    early_stopping = True
    model_name = "MOLIR"

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.hyperparameters = None

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Builds the model from hyperparameters.
        """
        self.hyperparameters = hyperparameters

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: Optional[FeatureDataset] = None,
        output_earlystopping: Optional[DrugResponseDataset] = None,
    ) -> None:
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
        if self.early_stopping and len(output_earlystopping) < 2:
            output_earlystopping = None
        dim_gex, dim_mut, dim_cnv = get_dimensions_of_omics_data(cell_line_input)
        self.model = MOLIModel(
            hpams=self.hyperparameters,
            input_dim_expr=dim_gex,
            input_dim_mut=dim_mut,
            input_dim_cnv=dim_cnv,
        )
        self.model.fit(
            output_train=output,
            cell_line_input=cell_line_input,
            output_earlystopping=output_earlystopping,
        )

    def predict(
        self,
        drug_ids: np.ndarray,
        cell_line_ids: np.ndarray,
        drug_input: FeatureDataset = None,
        cell_line_input: FeatureDataset = None,
    ) -> np.ndarray:
        input_data = self.get_feature_matrices(
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        gene_expression = input_data["gene_expression"]
        mutations = input_data["mutations"]
        cnvs = input_data["copy_number_variation_gistic"]
        return self.model.predict(gene_expression, mutations, cnvs)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        all_data = load_and_reduce_gene_features(
            feature_type="gene_expression",
            gene_list=None,
            data_path=data_path,
            dataset_name=dataset_name,
        )
        # log transformation
        all_data._apply(function=np.log, view="gene_expression")
        # in Toy_Data, everything is already in the dataset
        # TODO: implement this in models/utils.py
        mut_data = load_and_reduce_gene_features(
            feature_type="mutations",
            gene_list=None,
            data_path=data_path,
            dataset_name=dataset_name,
        )
        cnv_data = load_and_reduce_gene_features(
            feature_type="copy_number_variation_gistic",
            gene_list=None,
            data_path=data_path,
            dataset_name=dataset_name,
        )
        for fd in [mut_data, cnv_data]:
            all_data._add_features(fd)
        return all_data

    def load(self, path):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError
