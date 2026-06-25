"""Shared training utilities for literature model engines."""

from __future__ import annotations

import inspect
import os
from contextlib import suppress
from typing import Any

import numpy as np
import wandb
import yaml
from sklearn.model_selection import ParameterGrid

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.evaluation import AVAILABLE_METRICS


class LiteratureEngineBase:
    """Optional wandb/logging helpers and shared feature utilities for literature engines."""

    early_stopping = False
    is_single_drug_model = False
    cell_line_views: list[str]
    drug_views: list[str]

    def __init__(self) -> None:
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
        msg = f"{cls.__name__} must implement get_model_name"
        raise NotImplementedError(msg)

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        msg = f"{self.get_model_name()} must implement build_model"
        raise NotImplementedError(msg)

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
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
        _ = cell_line_ids, drug_ids, cell_line_input, drug_input
        msg = f"{self.get_model_name()} must implement predict"
        raise NotImplementedError(msg)

    @classmethod
    def get_hyperparameter_set(cls) -> list[dict[str, Any]]:
        hyperparameter_file = os.path.join(os.path.dirname(inspect.getfile(cls)), "hyperparameters.yaml")
        with open(hyperparameter_file, encoding="utf-8") as handle:
            try:
                hpams = yaml.safe_load(handle)[cls.get_model_name()]
            except yaml.YAMLError as exc:
                msg = f"Error in hyperparameters.yaml: {exc}"
                raise ValueError(msg) from exc
            except KeyError as exc:
                msg = f"Model {cls.get_model_name()} not found in hyperparameters.yaml"
                raise KeyError(msg) from exc
        if hpams is None:
            return [{}]
        for key, value in list(hpams.items()):
            if not isinstance(value, list):
                hpams[key] = [value]
        return list(ParameterGrid(hpams))

    def log_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        if not self.is_wandb_enabled():
            return
        self.hyperparameters = hyperparameters
        if not self._in_hyperparameter_tuning:
            wandb.config.update({"hyperparameters": hyperparameters})

    def is_wandb_enabled(self) -> bool:
        return self.wandb_project is not None and (self.wandb_run is not None or wandb.run is not None)

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        if not self.is_wandb_enabled():
            return
        with suppress(Exception):  # pragma: no cover - wandb may be unavailable
            wandb.log(metrics, step=step)

    def compute_performance_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        prefix: str = "",
    ) -> dict[str, float]:
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
        """Concatenate cell-line and drug views into a single feature matrix."""
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
