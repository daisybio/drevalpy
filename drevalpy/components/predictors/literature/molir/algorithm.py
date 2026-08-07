"""Contains the MOLIR model, a regression adaptation of the MOLI model.

Original authors: Sharifi-Noghabi et al. (2019, 10.1093/bioinformatics/btz318)
Code adapted from their Github: https://github.com/hosseinshn/MOLI
and Hauptmann et al. (2023, 10.1186/s12859-023-05166-7) https://github.com/kramerlab/Multi-Omics_analysis
"""

from pathlib import Path
from typing import Any

import numpy as np

from drevalpy.components.predictors.literature._training_helpers import LiteratureTrainingMixin
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from .utils import MOLIModel, filter_and_sort_omics, get_dimensions_of_omics_data


class MOLIR(LiteratureTrainingMixin):
    """Regression extension of MOLI: multi-omics late integration deep neural network.

    Takes somatic mutation, copy number variation and gene expression data as input. MOLI uses type-specific encoding
    subnetworks to learn features for each omics type, concatenates them into one representation and optimizes this
    representation via a combined cost function consisting of a triplet loss and a binary cross-entropy loss.
    We use a regression adaption with MSE loss and a mechanism to find positive and negative samples.
    """

    cell_line_views = ["gene_expression", "mutations", "copy_number_variation_gistic"]
    early_stopping = True
    is_single_drug_model = True

    def __init__(self) -> None:
        """Initializes the MOLIR model.

        The hyperparameters are set in configure, the model is set in train when we know the dimensionality of the
        gene expression, mutation and copy number variation data.
        """
        super().__init__()
        self.model: MOLIModel | None = None
        self.hyperparameters: dict[str, Any] = dict()
        self.gene_expression_features = None
        self.mutations_features = None
        self.copy_number_variation_features = None

    @classmethod
    def get_model_name(cls) -> str:
        """Returns the model name.

        :returns: returns: MOLIR
        """
        return "MOLIR"

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, Any]:
        """Return default MOLIR hyperparameters.

        :returns: Default hyperparameter mapping.
        """
        return {
            "mini_batch": 32,
            "h_dim1": 64,
            "h_dim2": 64,
            "h_dim3": 64,
            "learning_rate": 0.01,
            "dropout_rate": 0.5,
            "weight_decay": 0.0001,
            "gamma": 0.5,
            "epochs": 30,
            "margin": 1.5,
        }

    def configure(self, hyperparameters: dict[str, Any]) -> None:
        """Configure the model from hyperparameters.

        :param hyperparameters: Keys include ``mini_batch``, layer sizes, ``learning_rate``,
            ``dropout_rate``, ``weight_decay``, ``gamma``, ``epochs``, and ``margin``.
        """
        # Log hyperparameters to wandb if enabled
        self.log_hyperparameters(hyperparameters)

        self.hyperparameters = hyperparameters

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str | Path = "checkpoints",
    ) -> None:
        """Initializes and trains the model.

        First, the gene expression data was reduced using a variance threshold (0.05) and standardized. We chose to use
        the most variable 1000 genes instead to avoid issues with the variance threshold.
        Then, the model is initialized with the hyperparameters and the dimensions of the gene expression, mutation and
        copy number variation data. If there is no training data, the model is set to None (and predictions will be
        skipped as well). If there is not enough training data, the predictions will be made on the randomly
        initialized model.

        :param output: drug response data
        :param cell_line_input: cell line omics features, i.e., gene expression, mutations and copy number variation
        :param drug_input: drug features, not needed
        :param output_earlystopping: early stopping data, not used when there is not enough data
        :param model_checkpoint_dir: directory to save the model checkpoints
        """
        if len(output) > 0:
            self.gene_expression_features = cell_line_input.meta_info["gene_expression"]
            self.mutations_features = cell_line_input.meta_info["mutations"]
            self.copy_number_variation_features = cell_line_input.meta_info["copy_number_variation_gistic"]

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
                    model_checkpoint_dir=model_checkpoint_dir,
                    wandb_project=self.wandb_project,
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
        """Predicts the drug response.

        If there was no training data, only nans will be returned.

        :param cell_line_ids: Cell lines to predict
        :param drug_ids: Drugs to predict
        :param cell_line_input: cell line omics features
        :param drug_input: drug features, not needed

        :returns: returns: Predicted drug response

        :raises ValueError: If the model was not trained
        """
        if self.model is None:
            print("No model trained, will predict NA.")
            return np.array([np.nan] * len(cell_line_ids))
        if (
            (self.gene_expression_features is None)
            or (self.mutations_features is None)
            or (self.copy_number_variation_features is None)
        ):
            raise ValueError("MOLIR Model not trained, please train the model first.")

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

        gene_expression, mutations, cnvs = filter_and_sort_omics(
            model=self, gene_expression=gene_expression, mutations=mutations, cnvs=cnvs, cell_line_input=cell_line_input
        )
        return np.atleast_1d(self.model.predict(gene_expression, mutations, cnvs))
