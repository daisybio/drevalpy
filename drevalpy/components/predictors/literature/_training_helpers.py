"""Behavior-neutral training helpers for literature model algorithms."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import numpy as np
import wandb

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.evaluation import AVAILABLE_METRICS


class LiteratureTrainingMixin:
    """Shared FeatureDataset, wandb, and view-matrix helpers for literature algorithms."""

    early_stopping = False
    is_single_drug_model = False
    cell_line_views: list[str]
    drug_views: list[str]

    def __init__(self) -> None:
        """Initialize wandb bookkeeping and default view lists."""
        self.wandb_project: str | None = None
        self.wandb_run: Any = None
        self.wandb_config: dict[str, Any] | None = None
        self.hyperparameters: dict[str, Any] = {}
        self._in_hyperparameter_tuning: bool = False
        if not hasattr(self, "cell_line_views"):
            self.cell_line_views = []
        if not hasattr(self, "drug_views"):
            self.drug_views = []

    @classmethod
    def get_model_name(cls) -> str:
        """Return the registered model name for error messages.

        :raises NotImplementedError: Always raised by the mixin base implementation.
        """
        msg = f"{cls.__name__} must implement get_model_name"
        raise NotImplementedError(msg)

    def configure(self, hyperparameters: dict[str, Any]) -> None:
        """Apply hyperparameters to the algorithm (subclasses must override).

        :param hyperparameters: Algorithm hyperparameter mapping.

        :raises NotImplementedError: Always raised by the mixin base implementation.
        """
        msg = f"{self.get_model_name()} must implement configure"
        raise NotImplementedError(msg)

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """Train the algorithm on response and feature inputs (subclasses must override).

        :param output: Training responses and pair identifiers.
        :param cell_line_input: Cell-line feature dataset.
        :param drug_input: Optional drug feature dataset.
        :param output_earlystopping: Optional early-stopping responses.
        :param model_checkpoint_dir: Directory for model checkpoints.

        :raises NotImplementedError: Always raised by the mixin base implementation.
        """
        _ = output, cell_line_input, drug_input, output_earlystopping, model_checkpoint_dir
        msg = f"{self.get_model_name()} must implement train"
        raise NotImplementedError(msg)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """Predict responses for the requested pairs (subclasses must override).

        :param cell_line_ids: Cell-line identifiers in prediction order.
        :param drug_ids: Drug identifiers in prediction order.
        :param cell_line_input: Cell-line feature dataset.
        :param drug_input: Optional drug feature dataset.

        :raises NotImplementedError: Always raised by the mixin base implementation.
        """
        _ = cell_line_ids, drug_ids, cell_line_input, drug_input
        msg = f"{self.get_model_name()} must implement predict"
        raise NotImplementedError(msg)

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, Any]:
        """Return default hyperparameters for literature algorithms.

        :returns: Empty mapping in the mixin base implementation.
        """
        return {}

    def log_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        """Mirror hyperparameters to wandb when logging is enabled.

        :param hyperparameters: Hyperparameters to record for the active run.
        """
        if not self.is_wandb_enabled():
            return
        self.hyperparameters = hyperparameters
        if not self._in_hyperparameter_tuning:
            wandb.config.update({"hyperparameters": hyperparameters})

    def is_wandb_enabled(self) -> bool:
        """Return whether wandb logging is configured for this run.

        :returns: ``True`` when a wandb project and active run are available.
        """
        return self.wandb_project is not None and (self.wandb_run is not None or wandb.run is not None)

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        """Log evaluation metrics to wandb when enabled.

        :param metrics: Metric name to scalar value mapping.
        :param step: Optional global step for the wandb log entry.
        """
        if not self.is_wandb_enabled():
            return
        with suppress(Exception):
            wandb.log(metrics, step=step)

    def compute_performance_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        prefix: str = "",
    ) -> dict[str, float]:
        """Compute Pearson and R^2 metrics for predictions and targets.

        :param predictions: Model predictions.
        :param targets: Ground-truth response values.
        :param prefix: Optional prefix applied to returned metric names.

        :returns: Metric mapping, or an empty dict when computation fails.
        """
        try:
            metrics = {
                "R^2": AVAILABLE_METRICS["R^2"](y_pred=predictions, y_true=targets),
                "Pearson": AVAILABLE_METRICS["Pearson"](y_pred=predictions, y_true=targets),
            }
            if prefix:
                metrics = {f"{prefix}{key}": value for key, value in metrics.items()}
            return metrics
        except Exception:
            return {}

    def get_feature_matrices(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset | None,
        drug_input: FeatureDataset | None,
    ) -> dict[str, np.ndarray]:
        """Collect per-view feature matrices for the requested entity ids.

        :param cell_line_ids: Cell-line identifiers in batch order.
        :param drug_ids: Drug identifiers in batch order.
        :param cell_line_input: Optional cell-line feature dataset.
        :param drug_input: Optional drug feature dataset.

        :returns: Mapping from view name to feature matrix.

        :raises ValueError: If a configured view is missing from the input dataset.
        """
        cell_line_feature_matrices: dict[str, np.ndarray] = {}
        if cell_line_input is not None:
            for cell_line_view in self.cell_line_views:
                if cell_line_view not in cell_line_input.view_names:
                    msg = f"Cell line input does not contain view {cell_line_view}"
                    raise ValueError(msg)
                cell_line_feature_matrices[cell_line_view] = cell_line_input.get_feature_matrix(
                    view=cell_line_view,
                    identifiers=cell_line_ids,
                )
        drug_feature_matrices: dict[str, np.ndarray] = {}
        if drug_input is not None:
            for drug_view in self.drug_views:
                if drug_view not in drug_input.view_names:
                    msg = f"Drug input does not contain view {drug_view}"
                    raise ValueError(msg)
                drug_feature_matrices[drug_view] = drug_input.get_feature_matrix(
                    view=drug_view,
                    identifiers=drug_ids,
                )
        return {**cell_line_feature_matrices, **drug_feature_matrices}

    def get_concatenated_features(
        self,
        cell_line_view: str | None,
        drug_view: str | None,
        cell_line_ids_output: np.ndarray,
        drug_ids_output: np.ndarray,
        cell_line_input: FeatureDataset | None,
        drug_input: FeatureDataset | None,
    ) -> np.ndarray:
        """Concatenate cell-line and drug views into a single feature matrix.

        :param cell_line_view: Cell-line view to include, if any.
        :param drug_view: Drug view to include, if any.
        :param cell_line_ids_output: Cell-line ids in prediction order.
        :param drug_ids_output: Drug ids in prediction order.
        :param cell_line_input: Cell-line feature dataset.
        :param drug_input: Drug feature dataset.

        :returns: Concatenated feature matrix with one row per pair.

        :raises ValueError: If requested views are missing or no features are available.
        """
        inputs = self.get_feature_matrices(
            cell_line_ids=cell_line_ids_output,
            drug_ids=drug_ids_output,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        if drug_view is not None and drug_view not in inputs:
            msg = f"Expected drug_view '{drug_view}' to be in inputs, but it was not. Inputs: {inputs}"
            raise ValueError(msg)
        if cell_line_view is not None and cell_line_view not in inputs:
            msg = f"Expected cell_line_view '{cell_line_view}' to be in inputs, but it was not. Inputs: {inputs}"
            raise ValueError(msg)

        cell_line_features = None if cell_line_view is None else inputs.get(cell_line_view)
        drug_features = None if drug_view is None else inputs.get(drug_view)

        if cell_line_features is not None and drug_features is not None:
            return np.concatenate((cell_line_features, drug_features), axis=1)
        if cell_line_features is not None:
            return cell_line_features
        if drug_features is not None:
            return drug_features
        msg = "No features provided."
        raise ValueError(msg)
