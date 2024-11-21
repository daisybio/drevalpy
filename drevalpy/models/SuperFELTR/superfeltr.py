"""
Contains the SuperFELTR model.

Regression extension of Super.FELT: supervised feature extraction learning using triplet loss for drug response
prediction with multi-omics data.
Very similar to MOLI. Differences:

    * In MOLI, encoders and the classifier were trained jointly. Super.FELT trains them independently
    * MOLI was trained without feature selection (except for the Variance Threshold on the gene expression).
        Super.FELT uses feature selection for all omics data.

The input remains the same: somatic mutation, copy number variation and gene expression data.
Original authors of SuperFELT: Park, Soh & Lee. (2021, 10.1186/s12859-021-04146-z)
Code adapted from their Github: https://github.com/DMCB-GIST/Super.FELT
and Hauptmann et al. (2023, 10.1186/s12859-023-05166-7) https://github.com/kramerlab/Multi-Omics_analysis
"""

from typing import Any

import numpy as np
import pytorch_lightning as pl
from sklearn.feature_selection import VarianceThreshold

from ...datasets.dataset import DrugResponseDataset, FeatureDataset
from ..drp_model import DRPModel
from ..MOLIR.utils import get_dimensions_of_omics_data, make_ranges
from ..utils import get_multiomics_feature_dataset
from .utils import SuperFELTEncoder, SuperFELTRegressor, train_superfeltr_model


class SuperFELTR(DRPModel):
    """Regression extension of Super.FELT."""

    is_single_drug_model = True
    cell_line_views = ["gene_expression", "mutations", "copy_number_variation_gistic"]
    drug_views = []
    early_stopping = True

    def __init__(self) -> None:
        """
        Initialization method for SuperFELTR Model.

        The encoders and the regressor are initialized to None because they are built later in the first training pass.
        The hyperparameters are also initialized to an empty dict because they are initialized in build_model. The
        ranges are initialized during training which is why here, they get dummy values. The best checkpoint is
        determined after training.
        """
        super().__init__()
        # encoders and regressor are initialized to None because they are built later in the first training pass
        self.expr_encoder: SuperFELTEncoder | None = None
        self.mut_encoder: SuperFELTEncoder | None = None
        self.cnv_encoder: SuperFELTEncoder | None = None
        self.regressor: SuperFELTRegressor | None = None
        # hyperparameters are initialized to None because they are initialized in build_model
        self.hyperparameters: dict[str, Any] = dict()
        # ranges are initialized later because they are initialized using the standard variation of the train
        # response data which is only available when entering the training
        self.ranges: tuple[float, float] = (0.0, 1.0)
        # best checkpoint is determined after training
        self.best_checkpoint: pl.callbacks.ModelCheckpoint | None = None

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SuperFELTR
        """
        return "SuperFELTR"

    def build_model(self, hyperparameters) -> None:
        """
        Builds the model from hyperparameters.

        :param hyperparameters: dictionary containing the hyperparameters for the model. Contain mini_batch,
            dropout_rate, weight_decay, out_dim_expr_encoder, out_dim_mutation_encoder, out_dim_cnv_encoder, epochs,
            variance thresholds for gene expression, mutation, and copy number variation, margin, and learning rate.
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
        Does feature selection, trains the encoders sequentially, and then trains the regressor.

        If there is not enough training data, the model is trained with random initialization, if there is no
        training data at all, the model is skipped and later on, NA is predicted.

        :param output: training data associated with the response output
        :param cell_line_input: cell line omics features
        :param drug_input: not needed, as it is a single drug model
        :param output_earlystopping: optional early stopping dataset
        :raises ValueError: if drug_input is not None
        """
        if drug_input is not None:
            raise ValueError("SuperFELTR is a single drug model and does not require drug input.")

        if len(output) > 0:
            cell_line_input = self._feature_selection(output, cell_line_input)
            if output_earlystopping is not None and self.early_stopping and len(output_earlystopping) < 2:
                output_earlystopping = None
            dim_gex, dim_mut, dim_cnv = get_dimensions_of_omics_data(cell_line_input)
            self.ranges = make_ranges(output)

            # difference to MOLI: encoders and regressor are trained independently
            # Create and train encoders
            encoders = {}
            encoder_dims = {"expression": dim_gex, "mutation": dim_mut, "copy_number_variation_gistic": dim_cnv}
            for omic_type, dim in encoder_dims.items():
                encoder = SuperFELTEncoder(
                    input_size=dim, hpams=self.hyperparameters, omic_type=omic_type, ranges=self.ranges
                )
                if len(output) >= self.hyperparameters["mini_batch"]:
                    print(f"Training SuperFELTR Encoder for {omic_type} ... ")
                    best_checkpoint = train_superfeltr_model(
                        model=encoder,
                        hpams=self.hyperparameters,
                        output_train=output,
                        cell_line_input=cell_line_input,
                        output_earlystopping=output_earlystopping,
                        patience=5,
                    )
                    encoders[omic_type] = SuperFELTEncoder.load_from_checkpoint(best_checkpoint.best_model_path)
                else:
                    print(
                        f"Not enough training data provided for SuperFELTR Encoder for {omic_type}. Using random "
                        f"initialization."
                    )
                    encoders[omic_type] = encoder

            self.expr_encoder, self.mut_encoder, self.cnv_encoder = (
                encoders["expression"],
                encoders["mutation"],
                encoders["copy_number_variation_gistic"],
            )

            self.regressor = SuperFELTRegressor(
                input_size=self.hyperparameters["out_dim_expr_encoder"]
                + self.hyperparameters["out_dim_mutation_encoder"]
                + self.hyperparameters["out_dim_cnv_encoder"],
                hpams=self.hyperparameters,
                encoders=(self.expr_encoder, self.mut_encoder, self.cnv_encoder),
            )
            if len(output) >= self.hyperparameters["mini_batch"]:
                print("Training SuperFELTR Regressor ... ")
                self.best_checkpoint = train_superfeltr_model(
                    model=self.regressor,
                    hpams=self.hyperparameters,
                    output_train=output,
                    cell_line_input=cell_line_input,
                    output_earlystopping=output_earlystopping,
                    patience=5,
                )
            else:
                print("Not enough training data provided for SuperFELTR Regressor. Using random initialization.")
                self.best_checkpoint = None
        else:
            print("No training data provided, skipping model")
            self.best_checkpoint = None
            self.expr_encoder, self.mut_encoder, self.cnv_encoder, self.regressor = None, None, None, None

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the drug response.

        If there is no training data, NA is predicted. If there was not enough training data, predictions are made
        with the randomly initialized model.

        :param cell_line_ids: cell line ids
        :param drug_ids: drug ids
        :param cell_line_input: cell line omics features
        :param drug_input: drug omics features, not needed
        :returns: predicted drug response
        :raises ValueError: if drug_input is not None
        """
        if drug_input is not None:
            raise ValueError("SuperFELTR is a single drug model and does not require drug input.")

        input_data = self.get_feature_matrices(
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        gene_expression, mutations, cnvs = (
            input_data["gene_expression"],
            input_data["mutations"],
            input_data["copy_number_variation_gistic"],
        )
        if self.expr_encoder is None or self.mut_encoder is None or self.cnv_encoder is None or self.regressor is None:
            print("No training data was available, predicting NA")
            return np.array([np.nan] * len(cell_line_ids))
        if self.best_checkpoint is None:
            print("Not enough training data provided for SuperFELTR Regressor. Predicting with random initialization.")
            return self.regressor.predict(gene_expression, mutations, cnvs)
        best_regressor = SuperFELTRegressor.load_from_checkpoint(
            self.best_checkpoint.best_model_path,
            input_size=self.hyperparameters["out_dim_expr_encoder"]
            + self.hyperparameters["out_dim_mutation_encoder"]
            + self.hyperparameters["out_dim_cnv_encoder"],
            hpams=self.hyperparameters,
            encoders=(self.expr_encoder, self.mut_encoder, self.cnv_encoder),
        )
        return best_regressor.predict(gene_expression, mutations, cnvs)

    def _feature_selection(self, output: DrugResponseDataset, cell_line_input: FeatureDataset) -> FeatureDataset:
        """
        Feature selection for all omics data using the predefined variance thresholds.

        :param output: training data associated with the response output
        :param cell_line_input: cell line omics features
        :returns: cell line omics features with selected features
        """
        thresholds = {
            "gene_expression": self.hyperparameters["expression_var_threshold"][output.dataset_name],
            "mutations": self.hyperparameters["mutation_var_threshold"][output.dataset_name],
            "copy_number_variation_gistic": self.hyperparameters["cnv_var_threshold"][output.dataset_name],
        }
        for view in self.cell_line_views:
            selector = VarianceThreshold(thresholds[view])
            cell_line_input.fit_transform_features(
                train_ids=np.unique(output.cell_line_ids), transformer=selector, view=view
            )
        return cell_line_input

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features: gene expression, mutations, and copy number variation.

        :param data_path: path to the data, e.g., data/
        :param dataset_name: name of the dataset, e.g., GDSC2
        :returns: FeatureDataset containing the cell line gene expression features, mutations, and copy number variation
        """
        feature_dataset = get_multiomics_feature_dataset(
            data_path=data_path, dataset_name=dataset_name, gene_list=None, omics=self.cell_line_views
        )
        # log transformation
        feature_dataset.apply(function=np.log, view="gene_expression")
        return feature_dataset

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """
        Returns None, as drug features are not needed for SuperFELTR.

        :param data_path: Path to the fingerprints, e.g., data/
        :param dataset_name: Name of the dataset
        :returns: None
        """
        return None
