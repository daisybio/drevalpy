"""
Contains the MOLIR model, a regression adaptation of the MOLI model.

Original authors: Sharifi-Noghabi et al. (2019, 10.1093/bioinformatics/btz318)
Code adapted from their Github: https://github.com/hosseinshn/MOLI
and Hauptmann et al. (2023, 10.1186/s12859-023-05166-7) https://github.com/kramerlab/Multi-Omics_analysis
"""

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from ...datasets.dataset import DrugResponseDataset, FeatureDataset
from ..drp_model import DRPModel
from ..utils import VarianceFeatureSelector, get_multiomics_feature_dataset, scale_gene_expression
from .utils import MOLIModel, filter_and_sort_omics, get_dimensions_of_omics_data


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
        self.gene_expression_features = None
        self.mutations_features = None
        self.copy_number_variation_features = None
        self.gene_expression_scaler = StandardScaler()
        self.selector: VarianceFeatureSelector | None = None

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
        self.selector = VarianceFeatureSelector(
            view="gene_expression", k=hyperparameters.get("n_gene_expression_features", 1000)
        )

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Initializes and trains the model.

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
        :raises ValueError: If drug_input is None.
        """
        if len(output) > 0:
            cell_line_input = scale_gene_expression(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(output.cell_line_ids),
                training=True,
                gene_expression_scaler=self.gene_expression_scaler,
            )
            if self.selector is None:
                raise ValueError("Feature selector not initialized. Build the model first.")
            self.selector.fit(cell_line_input, output)
            cell_line_input = self.selector.transform(cell_line_input)

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

        cell_line_input = scale_gene_expression(
            cell_line_input=cell_line_input,
            cell_line_ids=np.unique(cell_line_ids),
            training=False,
            gene_expression_scaler=self.gene_expression_scaler,
        )
        # Apply variance threshold to gene expression features

        if self.selector is None:
            raise ValueError("Feature selector not initialized. Train the model first.")
        cell_line_input = self.selector.transform(cell_line_input)

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

        (gene_expression, mutations, cnvs) = filter_and_sort_omics(
            model=self, gene_expression=gene_expression, mutations=mutations, cnvs=cnvs, cell_line_input=cell_line_input
        )

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
            gene_lists={
                "gene_expression": "gene_expression_intersection",
                "mutations": "mutations_intersection",
                "copy_number_variation_gistic": "copy_number_variation_gistic_intersection",
            },
            omics=self.cell_line_views,
        )

        return feature_dataset

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """
        Returns None, as drug features are not needed for MOLIR.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: None
        """
        return None
