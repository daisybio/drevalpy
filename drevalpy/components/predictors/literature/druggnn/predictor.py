"""DrugGNN raw literature predictor."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.literature._algorithm_lifecycle import (
    predict_with_algorithm,
    train_fitted_algorithm,
)
from drevalpy.components.predictors.literature._metadata import DRUGGNN_REFERENCE
from drevalpy.components.predictors.literature._preload import (
    load_dataset_cell_line_features,
    load_dataset_drug_features,
)
from drevalpy.components.predictors.literature._raw_inputs import validate_raw_inputs
from drevalpy.components.predictors.literature._torch_state import load_object_mapping, save_object_mapping
from drevalpy.components.predictors.literature.druggnn.algorithm import DrugGNN
from drevalpy.components.predictors.literature.druggnn.state import apply_state, export_state
from drevalpy.components.predictors.raw_dataset import RawDatasetPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.registry import register_predictor
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.models.config import PredictionMode


@register_predictor(
    "drugGNN",
    description="DrugGNN: GCN on molecular graphs with dense cell-line features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.GRAPH,
    reference=DRUGGNN_REFERENCE,
)
class DrugGNNPredictor(RawDatasetPredictor):
    """Registered DrugGNN predictor."""

    supports_early_stopping: ClassVar[bool] = True
    required_cell_line_views: ClassVar[tuple[str, ...]] = ("gene_expression",)
    required_drug_views: ClassVar[tuple[str, ...]] = ("drug_graph",)
    requires_drug_featurizer: ClassVar[bool] = False
    validate_drug_graphs: ClassVar[bool] = True
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        super().__init__(hyperparameters)
        self._algorithm: DrugGNN | None = None
        self._engine_preload_state: dict[str, Any] = {}

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        defaults = dict(DrugGNN.get_default_hyperparameters())
        defaults.update({"epochs": 2, "batch_size": 8})
        return defaults

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {
            "hidden_dim": {"type": "int", "low": 16, "high": 128, "default": 64},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5, "default": 0.2},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True, "default": 1e-3},
            "epochs": {"type": "int", "low": 1, "high": 10, "default": 2},
            "batch_size": {"type": "int", "low": 4, "high": 32, "default": 8},
        }

    @classmethod
    def load_dataset_cell_line_features(
        cls,
        data_path: str,
        dataset_name: str,
        *,
        hyperparameters: dict[str, Any] | None = None,
        model_name: str | None = None,
    ) -> tuple[FeatureDataset, dict[str, Any]]:
        _ = model_name
        return load_dataset_cell_line_features(DrugGNN, data_path, dataset_name, hyperparameters=hyperparameters)

    @classmethod
    def load_dataset_drug_features(
        cls,
        data_path: str,
        dataset_name: str,
        *,
        hyperparameters: dict[str, Any] | None = None,
        model_name: str | None = None,
    ) -> tuple[FeatureDataset | None, dict[str, Any]]:
        _ = model_name
        features, preload = load_dataset_drug_features(
            DrugGNN, data_path, dataset_name, hyperparameters=hyperparameters
        )
        return features, preload

    def set_engine_preload_state(self, state: dict[str, Any]) -> None:
        self._engine_preload_state = dict(state)

    def _validated_inputs(self, batch: ModelInputBatch) -> tuple[FeatureDataset, FeatureDataset]:
        cell_lines, drugs = validate_raw_inputs(
            self,
            batch.cell_line_input,
            batch.drug_input,
            cell_line_views=self.required_cell_line_views,
            drug_views=self.required_drug_views,
            validate_drug_graphs=self.validate_drug_graphs,
        )
        return cell_lines, cast(FeatureDataset, drugs)

    def fit(self, batch: ModelInputBatch) -> None:
        cell_lines, drugs = self._validated_inputs(batch)
        self._algorithm = train_fitted_algorithm(
            DrugGNN,
            dict(self._hyperparameters),
            self._engine_preload_state,
            batch,
            cell_lines,
            drugs,
        )

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        cell_lines, drugs = self._validated_inputs(batch)
        return predict_with_algorithm(self._algorithm, batch, cell_lines, drugs)

    def is_fitted(self) -> bool:
        return self._algorithm is not None and self._algorithm.model is not None

    def get_state(self) -> dict[str, object]:
        if self._algorithm is None:
            return {}
        payload = export_state(self._algorithm)
        payload["predictor_hyperparameters"] = dict(self._hyperparameters)
        return {"payload": save_object_mapping(payload)}

    def set_state(self, state: dict[str, object]) -> None:
        blob = state.get("payload")
        if not isinstance(blob, (bytes, bytearray)):
            msg = f"{self.__class__.__name__} state requires a payload byte blob"
            raise PredictorStateError(msg)
        try:
            payload = load_object_mapping(bytes(blob))
        except Exception as exc:
            msg = f"{self.__class__.__name__} payload could not be deserialized"
            raise PredictorStateError(msg) from exc
        hyperparameters = payload.get("predictor_hyperparameters")
        if not isinstance(hyperparameters, dict):
            msg = f"{self.__class__.__name__} payload is missing predictor_hyperparameters"
            raise PredictorStateError(msg)
        if payload.get("model_state") is None:
            msg = f"{self.__class__.__name__} payload is missing a trained model"
            raise PredictorStateError(msg)
        self._hyperparameters = dict(hyperparameters)
        self._algorithm = apply_state(payload)
