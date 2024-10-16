"""
Contains the MOLI model.
Original authors: Sharifi-Noghabi et al. (2019, 10.1093/bioinformatics/btz318)
Code adapted from: Hauptmann et al. (2023, 10.1186/s12859-023-05166-7),
https://github.com/kramerlab/Multi-Omics_analysis
"""
from typing import Optional, Dict, Any

import numpy as np
from numpy._typing import ArrayLike
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from utils import Moli
from ..drp_model import DRPModel
from ..utils import load_and_reduce_gene_features
from ...datasets.dataset import FeatureDataset, DrugResponseDataset


class MOLIR(DRPModel):
    """
    Regression extension of
    MOLI: multi-omics late integration deep neural network.
    Takes somatic mutation, copy number variation and gene expression data as input.
    MOLI uses type-specific encoding sub-networks to learn features for each omics type,
    concatenates them into one representation and optimizes this representation via a combined cost
    function consisting of a triplet loss and a binary cross-entropy loss.
    We use a regression adaption with MSE loss and an mechanism to find positive and negative samples.
    """
    cell_line_views = [
        "gene_expression",
        "mutations",
        "copy_number_variation_gistic"
    ]
    drug_views = []
    early_stopping = True
    model_name = "MOLI"

    def __init__(self):
        super().__init__()
        self.model = None
        self.selector_gex = None
        self.scaler_gex = None

    def build_model(self, hyperparameters: Dict[str, Any]):
        """
        Builds the model from hyperparameters.
        """
        self.model = Moli(hpams=hyperparameters)

    def train(self, output: DrugResponseDataset, cell_line_input: FeatureDataset,
              drug_input: Optional[FeatureDataset] = None,
              output_earlystopping: Optional[DrugResponseDataset] = None) -> None:
        self.selector_gex = VarianceThreshold(0.05)
        self.selector_gex = cell_line_input.fit_transform_features(
            train_ids=np.unique(output.cell_line_ids),
            transformer=self.selector_gex,
            view="gene_expression"
        )
        self.scaler_gex = StandardScaler()
        self.scaler_gex = cell_line_input.fit_transform_features(
            train_ids=np.unique(output.cell_line_ids),
            transformer=self.scaler_gex,
            view="gene_expression"
        )
        # TODO
        self.model.fit(

        )



    def predict(self, drug_ids: ArrayLike, cell_line_ids: ArrayLike,
                drug_input: FeatureDataset = None,
                cell_line_input: FeatureDataset = None) -> np.ndarray:
        # TODO
        pass

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        all_data = load_and_reduce_gene_features(
            feature_type="gene_expression",
            gene_list=None,
            data_path=data_path,
            dataset_name=dataset_name
        )
        # log transformation
        all_data.apply(function=np.log, view="gene_expression")
        mut_data = load_and_reduce_gene_features(
            feature_type="mutations",
            gene_list=None,
            data_path=data_path,
            dataset_name=dataset_name
        )
        cnv_data = load_and_reduce_gene_features(
            feature_type="copy_number_variation_gistic",
            gene_list=None,
            data_path=data_path,
            dataset_name=dataset_name
        )
        for fd in [mut_data, cnv_data]:
            all_data.add_features(fd)
        return all_data

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        pass
