"""Contains the baseline MultiOmicsNeuralNetwork model."""

import warnings

import numpy as np
from sklearn.decomposition import PCA

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from ..drp_model import DRPModel
from ..utils import get_multiomics_feature_dataset, load_drug_fingerprint_features
from .utils import FeedForwardNetwork


class MultiOmicsNeuralNetwork(DRPModel):
    """Simple Feedforward Neural Network model with dropout using multiple omics data."""

    cell_line_views = [
        "gene_expression",
        "methylation",
        "mutations",
        "copy_number_variation_gistic",
    ]
    drug_views = ["fingerprints"]
    early_stopping = True

    def __init__(self):
        """
        Initalization method for MultiOmicsNeuralNetwork Model.

        The model and the PCA are initialized to None because they are built later in the build_model method.
        """
        super().__init__()
        self.model = None
        self.hyperparameters = None
        self.pca = None

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: MultiOmicsNeuralNetwork
        """
        return "MultiOmicsNeuralNetwork"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.

        The model is a simple feedforward neural network with dropout. The PCA is used to reduce the dimensionality of
        the methylation data.

        :param hyperparameters: dictionary containing the hyperparameters units_per_layer, dropout_prob, and
            methylation_pca_components.
        """
        self.hyperparameters = hyperparameters
        self.pca = PCA(n_components=hyperparameters["methylation_pca_components"])

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "",
    ):
        """
        Fits the PCA and trains the model.

        :param output: training data associated with the response output
        :param cell_line_input: cell line omics features
        :param drug_input: drug omics features
        :param output_earlystopping: optional early stopping dataset
        :param model_checkpoint_dir: directory to save the model checkpoints
        :raises ValueError: if drug_input (fingerprints) is missing
        """
        if drug_input is None:
            raise ValueError("Drug input (fingerprints) is needed for the MultiOmicsNeuralNetwork model.")

        unique_methylation = np.stack(
            [cell_line_input.features[id_]["methylation"] for id_ in np.unique(output.cell_line_ids)],
            axis=0,
        )

        self.pca.n_components = min(self.pca.n_components, len(unique_methylation))
        self.pca = self.pca.fit(unique_methylation)

        first_feature = next(iter(cell_line_input.features.values()))
        dim_gex = first_feature["gene_expression"].shape[0]
        dim_met = self.pca.n_components
        dim_mut = first_feature["mutations"].shape[0]
        dim_cnv = first_feature["copy_number_variation_gistic"].shape[0]
        dim_fingerprint = next(iter(drug_input.features.values()))["fingerprints"].shape[0]

        self.model = FeedForwardNetwork(
            hyperparameters=self.hyperparameters,
            input_dim=dim_gex + dim_met + dim_mut + dim_cnv + dim_fingerprint,
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
                trainer_params={
                    "max_epochs": self.hyperparameters.get("max_epochs", 100),
                    "progress_bar_refresh_rate": 500,
                },
                batch_size=16,
                patience=5,
                num_workers=1,
                met_transform=self.pca,
                model_checkpoint_dir=model_checkpoint_dir,
            )

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Transforms the methylation data using the fitted PCA and then predicts the response for the given input.

        :param drug_ids: drug identifiers
        :param cell_line_ids: cell line identifiers
        :param drug_input: drug omics features
        :param cell_line_input: cell line omics features
        :returns: predicted response
        :raises ValueError: if no cell line information was added
        """
        inputs = self.get_feature_matrices(
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        x: np.ndarray | None = None
        for cl_view in self.cell_line_views:
            feature_mat = inputs[cl_view]
            if cl_view == "methylation":
                feature_mat = self.pca.transform(feature_mat)
            if x is None:
                x = feature_mat
            else:
                x = np.concatenate((x, feature_mat), axis=1)
        for d_view in self.drug_views:
            feature_mat = inputs[d_view]
            if x is None:
                raise ValueError("There should be cell line features!")
            x = np.concatenate((x, feature_mat), axis=1)
        return self.model.predict(x)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features.

        :param data_path: data path e.g. data/
        :param dataset_name: dataset name e.g. GDSC1
        :return: FeatureDataset containing the cell line omics features, filtered through the
            drug target genes
        """
        gene_lists = {
            "gene_expression": "drug_target_genes_all_drugs",
            "methylation": "methylation_intersection",
            "mutations": "drug_target_genes_all_drugs",
            "copy_number_variation_gistic": "drug_target_genes_all_drugs",
            "proteomics": "drug_target_genes_all_drugs_proteomics",
        }
        return get_multiomics_feature_dataset(data_path=data_path, gene_lists=gene_lists, dataset_name=dataset_name)

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load the drug features.

        :param data_path: path to the drug features, in this case the drug fingerprints, e.g., data/
        :param dataset_name: name of the dataset, e.g., GDSC1
        :returns: FeatureDataset containing the drug fingerprint features
        """
        return load_drug_fingerprint_features(data_path, dataset_name, fill_na=True)
