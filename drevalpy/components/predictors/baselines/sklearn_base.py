"""Shared public adapters for sklearn predictors."""

from __future__ import annotations

from typing import Self

import numpy as np

from drevalpy.components.register_builtins import ensure_components_registered
from drevalpy.data.features import _get_view_as_list, load_single_cell_line_view, load_single_drug_view
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models._component_bridge import (
    ComponentDRPBridge,
    preview_sklearn_estimator,
    save_component_stack,
)
from drevalpy.models._legacy_checkpoint_loaders import (
    has_component_stack,
    load_legacy_sklearn_checkpoint,
    load_native_checkpoint,
)
from drevalpy.models._legacy_state_accessors import (
    sklearn_estimator_from_bridge,
    sklearn_featurizer_state_from_bridge,
)
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.factory import SKLEARN_PREDICTOR_BY_MODEL_NAME, sklearn_model_config
from drevalpy.models.single_drug import SingleDrugModelMixin


class SklearnModel(DRPModel):
    """Common DRPModel adapter for component-backed sklearn predictors."""

    cell_line_views = []
    drug_views = []

    def __init__(self) -> None:
        """Initialize component state and preprocessing defaults."""
        super().__init__()
        self._component_bridge = ComponentDRPBridge()
        self._preview_model = None
        self.proteomics_feature_threshold = 0.7
        self.proteomics_n_features = 1000
        self.proteomics_normalization_width = 0.3
        self.proteomics_normalization_downshift = 1.8
        self.methylation_n_components = 100

    @property
    def model(self):
        """Return the fitted estimator or the unfitted preview estimator."""
        fitted = sklearn_estimator_from_bridge(self._component_bridge)
        if fitted is not None:
            return fitted
        return self._preview_model

    @property
    def gene_expression_scaler(self):
        """Return the legacy gene-expression scaler view."""
        return sklearn_featurizer_state_from_bridge(self._component_bridge).get("gene_expression_scaler")

    @property
    def methylation_scaler(self):
        """Return the legacy methylation scaler view."""
        return sklearn_featurizer_state_from_bridge(self._component_bridge).get("methylation_scaler")

    @property
    def methylation_pca(self):
        """Return the legacy methylation PCA view."""
        return sklearn_featurizer_state_from_bridge(self._component_bridge).get("methylation_pca")

    @property
    def proteomics_transformer(self):
        """Return the legacy proteomics transformer view."""
        return sklearn_featurizer_state_from_bridge(self._component_bridge).get("proteomics_transformer")

    @classmethod
    def get_model_name(cls) -> str:
        """Return the public model name implemented by a concrete adapter."""
        raise NotImplementedError("get_model_name method has to be implemented in the child class.")

    def _drug_views_from_hyperparameters(self, hyperparameters: dict) -> list[str]:
        """Resolve the required drug views for a multi-drug model."""
        views = _get_view_as_list(hyperparameters.get("drug_views", ["fingerprints"]))
        if not views:
            msg = "Multi-drug sklearn models require at least one drug view"
            raise ValueError(msg)
        return views

    def build_model(self, hyperparameters: dict) -> None:
        """Build the configured component-backed sklearn model."""
        self.log_hyperparameters(hyperparameters)
        self.hyperparameters = hyperparameters
        self.cell_line_views = _get_view_as_list(hyperparameters.get("cell_line_views", ["gene_expression"]))
        self.drug_views = self._drug_views_from_hyperparameters(hyperparameters)

        if "proteomics" in self.cell_line_views:
            self._init_proteomics_features(hyperparameters)
        if "methylation" in self.cell_line_views:
            self.methylation_n_components = hyperparameters.get("methylation_n_components", 100)
        self._init_component_model()

    def _normalize_hyperparameters(self, hyperparameters: dict) -> dict:
        normalized = dict(hyperparameters)
        if normalized.get("max_depth") == "None":
            normalized["max_depth"] = None
        return normalized

    def _init_component_model(self) -> None:
        ensure_components_registered()
        predictor_type = SKLEARN_PREDICTOR_BY_MODEL_NAME[self.get_model_name()]
        hp = self._normalize_hyperparameters(self.hyperparameters)
        hp["cell_line_views"] = self.cell_line_views
        hp["drug_views"] = self.drug_views
        config = sklearn_model_config(predictor_type, hp)
        self._component_bridge.set_composed_config(config)
        self._preview_model = preview_sklearn_estimator(self._component_bridge, hp)

    def _init_proteomics_features(self, hyperparameters: dict) -> None:
        self.proteomics_feature_threshold = hyperparameters.get("proteomics_feature_threshold", 0.7)
        self.proteomics_n_features = hyperparameters.get("proteomics_n_features", 1000)
        self.proteomics_normalization_width = hyperparameters.get("proteomics_normalization_width", 0.3)
        self.proteomics_normalization_downshift = hyperparameters.get("proteomics_normalization_downshift", 1.8)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load the configured cell-line views."""
        return load_single_cell_line_view(self.cell_line_views, data_path, dataset_name, self.get_model_name())

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """Load the configured drug views."""
        return load_single_drug_view(self.drug_views, data_path, dataset_name, self.get_model_name())

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """Train the sklearn predictor through the component bridge."""
        _ = output_earlystopping, model_checkpoint_dir
        if len(output) == 0:
            print("No training data provided, will predict NA.")
            self._preview_model = None
            return
        self._component_bridge.train(output, cell_line_input, drug_input)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """Predict through the component bridge."""
        if not self._component_bridge.is_trained():
            print("No training data was available, predicting NA.")
            return np.array([np.nan] * len(cell_line_ids))
        return self._component_bridge.predict(cell_line_ids, drug_ids, cell_line_input, drug_input)

    def save(self, directory: str) -> None:
        """Save the trained component stack."""
        if not self._component_bridge.is_trained():
            raise ValueError("Cannot save: model is not trained.")
        save_component_stack(
            self._component_bridge,
            directory,
            hyperparameters=getattr(self, "hyperparameters", {}),
        )

    @classmethod
    def load(cls, directory: str) -> Self:
        """Load a native or legacy sklearn checkpoint."""
        instance = cls()
        if has_component_stack(directory):
            load_native_checkpoint(instance, directory)
        else:
            load_legacy_sklearn_checkpoint(instance, directory)
        return instance


class SingleDrugSklearnModel(SingleDrugModelMixin, SklearnModel):
    """Sklearn adapter trained independently per drug without drug features."""

    early_stopping = False

    def _drug_views_from_hyperparameters(self, hyperparameters: dict) -> list[str]:
        _ = hyperparameters
        return []

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """Return no drug features because each fitted model sees one drug."""
        _ = data_path, dataset_name
        return None
