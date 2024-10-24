"""
Contains the SuperFELTR model.
Original authors of SuperFELT: Park, Soh & Lee. (2021, 10.1186/s12859-021-04146-z)
Code adapted from their Github: https://github.com/DMCB-GIST/Super.FELT
and Hauptmann et al. (2023, 10.1186/s12859-023-05166-7) https://github.com/kramerlab/Multi-Omics_analysis
"""
from typing import Optional
import numpy as np
import torch
from sklearn.feature_selection import VarianceThreshold

from .utils import SuperFELTEncoder, SuperFELTRegressor, train_superfeltr_model
from ..MOLIR.utils import make_ranges, get_dimensions_of_omics_data
from ..drp_model import SingleDrugModel
from ..utils import load_and_reduce_gene_features
from ...datasets.dataset import FeatureDataset, DrugResponseDataset


class SuperFELTR(SingleDrugModel):
    """
    Regression extension of Super.FELT: supervised feature extraction learning using triplet loss for drug response
    prediction with multi-omics data.
    Very similar to MOLI. Differences:
    - In MOLI, encoders and the classifier were trained jointly. Super.FELT trains them independently
    - MOLI was trained without feature selection (except for the Variance Threshold on the gene expression).
    Super.FELT uses feature selection for all omics data.
    The input remains the same: somatic mutation, copy number variation and gene expression data.
    """
    cell_line_views = ["gene_expression", "mutations", "copy_number_variation_gistic"]
    drug_views = []
    early_stopping = True
    model_name = "SuperFELTR"

    def __init__(self):
        super().__init__()
        self.expr_encoder = None
        self.mut_encoder = None
        self.cnv_encoder = None
        self.regressor = None
        self.hyperparameters = None
        self.ranges = None
        self.best_checkpoint = None

    def build_model(self, hyperparameters):
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
        """
        Trains the model.
        """
        cell_line_input = self.feature_selection(output, cell_line_input)
        if self.early_stopping and len(output_earlystopping) < 2:
            output_earlystopping = None
        dim_gex, dim_mut, dim_cnv = get_dimensions_of_omics_data(cell_line_input)
        self.ranges = make_ranges(output)

        # difference to MOLI: encoders and regressor are trained independently
        self.expr_encoder = SuperFELTEncoder(
            input_size=dim_gex,
            hpams=self.hyperparameters,
            omic_type='expression',
            ranges=self.ranges
        )
        self.mut_encoder = SuperFELTEncoder(
            input_size=dim_mut,
            hpams=self.hyperparameters,
            omic_type='mutation',
            ranges=self.ranges
        )
        self.cnv_encoder = SuperFELTEncoder(
            input_size=dim_cnv,
            hpams=self.hyperparameters,
            omic_type='copy_number_variation_gistic',
            ranges=self.ranges
        )
        checkpoints = {}
        for encoder in [self.expr_encoder, self.mut_encoder, self.cnv_encoder]:
            print(f"Training SuperFELTR Encoder for {encoder.omic_type} ... ")
            best_checkpoint = train_superfeltr_model(
                model=encoder,
                hpams=self.hyperparameters,
                output_train=output,
                cell_line_input=cell_line_input,
                output_earlystopping=output_earlystopping,
                patience=5
            )
            checkpoints[encoder.omic_type] = best_checkpoint
        self.expr_encoder = SuperFELTEncoder.load_from_checkpoint(checkpoints["expression"].best_model_path,
                                                                  input_size=dim_gex,
                                                                  hpams=self.hyperparameters,
                                                                  omic_type='expression',
                                                                  ranges=self.ranges)
        self.mut_encoder = SuperFELTEncoder.load_from_checkpoint(checkpoints["mutation"].best_model_path,
                                                                 input_size=dim_mut,
                                                                 hpams=self.hyperparameters,
                                                                 omic_type='mutation',
                                                                 ranges=self.ranges
                                                                 )
        self.cnv_encoder = SuperFELTEncoder.load_from_checkpoint(checkpoints[
                                                                     "copy_number_variation_gistic"].best_model_path,
                                                                 input_size=dim_cnv,
                                                                 hpams=self.hyperparameters,
                                                                 omic_type='copy_number_variation_gistic',
                                                                 ranges=self.ranges
                                                                 )

        self.regressor = SuperFELTRegressor(
            input_size=self.hyperparameters["out_dim_expr_encoder"] + self.hyperparameters[
                "out_dim_mutation_encoder"] + self.hyperparameters["out_dim_cnv_encoder"],
            hpams=self.hyperparameters,
            encoders=[self.expr_encoder, self.mut_encoder, self.cnv_encoder],
            ranges=self.ranges
        )
        self.best_checkpoint = train_superfeltr_model(
            model=self.regressor,
            hpams=self.hyperparameters,
            output_train=output,
            cell_line_input=cell_line_input,
            output_earlystopping=output_earlystopping,
            patience=5
        )

    def predict(
            self,
            drug_ids: np.ndarray,
            cell_line_ids: np.ndarray,
            drug_input: FeatureDataset = None,
            cell_line_input: FeatureDataset = None,
    ) -> np.ndarray:
        """
        Predicts the drug response.
        """
        input_data = self.get_feature_matrices(
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        gene_expression = input_data["gene_expression"]
        mutations = input_data["mutations"]
        cnvs = input_data["copy_number_variation_gistic"]
        best_regressor = SuperFELTRegressor.load_from_checkpoint(self.best_checkpoint.best_model_path,
                                                                 input_size=self.hyperparameters[
                                                                                "out_dim_expr_encoder"] +
                                                                            self.hyperparameters[
                                                                                "out_dim_mutation_encoder"] +
                                                                            self.hyperparameters["out_dim_cnv_encoder"],
                                                                 hpams=self.hyperparameters,
                                                                 encoders=[self.expr_encoder, self.mut_encoder,
                                                                           self.cnv_encoder],
                                                                 ranges=self.ranges
                                                                 )
        return best_regressor.predict(gene_expression, mutations, cnvs)

    def feature_selection(self, output: DrugResponseDataset, cell_line_input: FeatureDataset) -> FeatureDataset:
        """
        Feature selection for all omics data.
        """
        gex_var_threshold = self.hyperparameters["expression_var_threshold"][output.dataset_name]
        mut_var_threshold = self.hyperparameters["mutation_var_threshold"][output.dataset_name]
        cnv_var_threshold = self.hyperparameters["cnv_var_threshold"][output.dataset_name]
        for view in self.cell_line_views:
            if view == "gene_expression":
                selector = VarianceThreshold(gex_var_threshold)
            elif view == "mutations":
                selector = VarianceThreshold(mut_var_threshold)
            elif view == "copy_number_variation_gistic":
                selector = VarianceThreshold(cnv_var_threshold)
            else:
                raise ValueError(f"Unknown view {view}")
            cell_line_input.fit_transform_features(
                train_ids=np.unique(output.cell_line_ids),
                transformer=selector,
                view=view,
            )
        return cell_line_input

    def load_cell_line_features(
        self, data_path: str, dataset_name: str
    ) -> FeatureDataset:
        all_data = load_and_reduce_gene_features(
            feature_type="gene_expression",
            gene_list=None,
            data_path=data_path,
            dataset_name=dataset_name,
        )
        # log transformation
        all_data._apply(function=np.log, view="gene_expression")
        if dataset_name != "Toy_Data":
            # in Toy_Data, everything is already in the dataset
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
