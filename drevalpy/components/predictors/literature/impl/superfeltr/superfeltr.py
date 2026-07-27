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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .utils import SuperFELTEncoder, SuperFELTRegressor

import numpy as np
import pytorch_lightning as pl

from drevalpy.components.predictors.literature._engine_base import LiteratureEngineBase
from drevalpy.components.predictors.literature.impl.molir.utils import filter_and_sort_omics
from drevalpy.data.features import get_multiomics_feature_dataset
from drevalpy.data.preprocessing import VarianceFeatureSelector
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from .training import run_superfeltr_training


class SuperFELTR(LiteratureEngineBase):
    """Regression extension of Super.FELT."""

    cell_line_views = ["gene_expression", "mutations", "copy_number_variation_gistic"]
    early_stopping = True
    is_single_drug_model = True

    def __init__(self) -> None:
        """
        Initialization method for SuperFELTR Model.

        The encoders and the regressor are initialized to None because they are built later in the first training pass.
        The hyperparameters are also initialized to an empty dict because they are initialized in configure. The
        ranges are initialized during training which is why here, they get dummy values. The best checkpoint is
        determined after training.
        """
        super().__init__()
        # encoders and regressor are initialized to None because they are built later in the first training pass
        self.expr_encoder: SuperFELTEncoder | None = None
        self.mut_encoder: SuperFELTEncoder | None = None
        self.cnv_encoder: SuperFELTEncoder | None = None
        self.regressor: SuperFELTRegressor | None = None
        self.hyperparameters: dict[str, Any] = dict()
        # ranges are initialized later because they are initialized using the standard variation of the train
        # response data which is only available when entering the training
        self.ranges: tuple[float, float] = (0.0, 1.0)
        # best checkpoint is determined after training
        self.best_checkpoint: pl.callbacks.ModelCheckpoint | None = None
        self.gene_expression_features = None
        self.mutations_features = None
        self.copy_number_variation_features = None
        self.selectors: dict[str, VarianceFeatureSelector] = {}

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SuperFELTR
        """
        return "SuperFELTR"

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        return {
            "mini_batch": 55,
            "dropout_rate": 0.5,
            "weight_decay": 0.01,
            "out_dim_expr_encoder": 256,
            "out_dim_mutation_encoder": 32,
            "out_dim_cnv_encoder": 64,
            "epochs": 30,
            "margin": 1.0,
            "learning_rate": 0.01,
        }

    def configure(self, hyperparameters) -> None:
        """
        Configure the model from hyperparameters.

        :param hyperparameters: dictionary containing the hyperparameters for the model. Contain mini_batch,
            dropout_rate, weight_decay, out_dim_expr_encoder, out_dim_mutation_encoder, out_dim_cnv_encoder, epochs,
            variance thresholds for gene expression, mutation, and copy number variation, margin, and learning rate.
        """
        # Log hyperparameters to wandb if enabled
        self.log_hyperparameters(hyperparameters)

        self.hyperparameters = hyperparameters

        n_features = hyperparameters.get("n_features_per_view", 1000)
        for view in self.cell_line_views:
            self.selectors[view] = VarianceFeatureSelector(view=view, k=n_features)

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "superfeltr_checkpoints",
    ) -> None:
        """
        Does feature selection, trains the encoders sequentially, and then trains the regressor.

        If there is not enough training data, the model is trained with random initialization, if there is no
        training data at all, the model is skipped and later on, NA is predicted.

        :param output: training data associated with the response output
        :param cell_line_input: cell line omics features
        :param drug_input: not needed, as it is a single drug model
        :param output_earlystopping: optional early stopping dataset
        :param model_checkpoint_dir: not needed
        :raises ValueError: if drug_input is not None
        """
        if drug_input is not None:
            raise ValueError("SuperFELTR is a single drug model and does not require drug input.")

        run_superfeltr_training(
            self,
            output=output,
            cell_line_input=cell_line_input,
            output_earlystopping=output_earlystopping,
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
        if self.expr_encoder is None or self.mut_encoder is None or self.cnv_encoder is None or self.regressor is None:
            print("No training data was available, predicting NA")
            return np.array([np.nan] * len(cell_line_ids))
        if (
            self.gene_expression_features is None
            or self.mutations_features is None
            or self.copy_number_variation_features is None
        ):
            raise ValueError("Model was not trained, no features available.")

        if drug_input is not None:
            raise ValueError("SuperFELTR is a single drug model and does not require drug input.")

        for view in self.cell_line_views:
            selector = self.selectors[view]
            cell_line_input = selector.transform(cell_line_input)

        input_data = self.get_feature_matrices(
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )

        gene_expression = input_data["gene_expression"]
        mutations = input_data["mutations"]
        cnvs = input_data["copy_number_variation_gistic"]

        gene_expression, mutations, cnvs = filter_and_sort_omics(
            model=self, gene_expression=gene_expression, mutations=mutations, cnvs=cnvs, cell_line_input=cell_line_input
        )

        if self.best_checkpoint is None:
            print("Not enough training data provided for SuperFELTR Regressor. Predicting with random initialization.")
            return self.regressor.predict(gene_expression, mutations, cnvs)

        return self.regressor.predict(gene_expression, mutations, cnvs)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features: gene expression, mutations, and copy number variation.

        :param data_path: path to the data, e.g., data/
        :param dataset_name: name of the dataset, e.g., GDSC2
        :returns: FeatureDataset containing the cell line gene expression features, mutations, and copy number variation
        """
        feature_dataset = get_multiomics_feature_dataset(
            data_path=data_path, dataset_name=dataset_name, gene_lists=None, omics=self.cell_line_views
        )
        # log transformation
        feature_dataset.apply(function=np.arcsinh, view="gene_expression")
        return feature_dataset

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """
        Returns None, as drug features are not needed for SuperFELTR.

        :param data_path: Path to the fingerprints, e.g., data/
        :param dataset_name: Name of the dataset
        :returns: None
        """
        return None

    def _fit_feature_selection(self, output: DrugResponseDataset, cell_line_input: FeatureDataset) -> FeatureDataset:
        for view in self.cell_line_views:
            selector = self.selectors[view]
            selector.fit(cell_line_input, output)
            cell_line_input = selector.transform(cell_line_input)

        self.gene_expression_features = cell_line_input.meta_info["gene_expression"]
        self.mutations_features = cell_line_input.meta_info["mutations"]
        self.copy_number_variation_features = cell_line_input.meta_info["copy_number_variation_gistic"]
        return cell_line_input
