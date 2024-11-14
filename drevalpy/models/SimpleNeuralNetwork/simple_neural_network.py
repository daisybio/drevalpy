"""Contains the SimpleNeuralNetwork model."""

import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from ..drp_model import DRPModel
from ..utils import load_and_reduce_gene_features, load_drug_fingerprint_features
from .utils import FeedForwardNetwork


class SimpleNeuralNetwork(DRPModel):
    """Simple Feedforward Neural Network model with dropout using only gene expression data."""

    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]
    early_stopping = True

    def __init__(self):
        """Initializes the SimpleNeuralNetwork.

        The model is build in train(). The gene_expression_scalar is set to the StandardScaler() and later fitted
        using the training data only.
        """
        super().__init__()
        self.model = None
        self.hyperparameters = None
        self.gene_expression_scaler = StandardScaler()

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SimpleNeuralNetwork
        """
        return "SimpleNeuralNetwork"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.

        :param hyperparameters: includes units_per_layer and dropout_prob.
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
        First scales the gene expression data and trains the model.

        The gene expression data is first arcsinh transformed. Afterward, the StandardScaler() is fitted on the
        training gene expression data only. Then, it transforms all gene expression data.
        :param output: training data associated with the response output
        :param cell_line_input: cell line omics features
        :param drug_input: drug omics features
        :param output_earlystopping: optional early stopping dataset
        :raises ValueError: if drug_input (fingerprints) is missing

        """
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) are required for SimpleNeuralNetwork.")

        # Apply arcsinh transformation and scaling to gene expression features
        if "gene_expression" in self.cell_line_views:
            cell_line_input.apply(function=np.arcsinh, view="gene_expression")
            self.gene_expression_scaler = cell_line_input.fit_transform_features(
                train_ids=np.unique(output.cell_line_ids),
                transformer=self.gene_expression_scaler,
                view="gene_expression",
            )

        dim_gex = next(iter(cell_line_input.features.values()))["gene_expression"].shape[0]
        dim_fingerprint = next(iter(drug_input.features.values()))["fingerprints"].shape[0]

        self.model = FeedForwardNetwork(
            hyperparameters=self.hyperparameters,
            input_dim=dim_gex + dim_fingerprint,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*does not have many workers which may be a bottleneck.*",
            )
            self.model.fit(
                output_train=output,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
                cell_line_views=self.cell_line_views,
                drug_views=self.drug_views,
                output_earlystopping=output_earlystopping,
                batch_size=16,
                patience=5,
                num_workers=1,
            )

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the response for the given input.

        :param cell_line_ids: IDs of the cell lines to be predicted
        :param drug_ids: IDs of the drugs to be predicted
        :param cell_line_input: gene expression of the test data
        :param drug_input: fingerprints of the test data
        :returns: the predicted drug responses
        """
        x = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view="fingerprints",
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )

        return self.model.predict(x)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features.

        :param data_path: Path to the gene expression and landmark genes
        :param dataset_name: name of the dataset
        :return: FeatureDataset containing the cell line gene expression features, filtered through the landmark genes
        """
        return load_and_reduce_gene_features(
            feature_type="gene_expression",
            gene_list="landmark_genes",
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the fingerprint data.

        :param data_path: Path to the fingerprints, e.g., data/
        :param dataset_name: name of the dataset, e.g., GDSC1
        :returns: FeatureDataset containing the fingerprints
        """
        return load_drug_fingerprint_features(data_path, dataset_name)
