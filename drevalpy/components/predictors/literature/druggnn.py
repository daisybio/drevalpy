"""DrugGNN structured literature predictor."""

from __future__ import annotations

import io
from typing import Any, ClassVar

import joblib
import numpy as np

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.literature._metadata import DRUGGNN_METADATA
from drevalpy.components.predictors.literature.impl.druggnn.drug_gnn import DrugGNN as DrugGNNEngine
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.predictors.structured import StructuredPredictor
from drevalpy.components.registry import register_predictor
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset


@register_predictor(
    "drugGNN",
    description="DrugGNN: GCN on molecular graphs with dense cell-line features.",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.GRAPH,
    **DRUGGNN_METADATA,
)
class DrugGNNPredictor(StructuredPredictor):
    """DrugGNN predictor component backed by the parity-checked engine."""

    supports_early_stopping: ClassVar[bool] = True

    def __init__(self) -> None:
        self._hyperparameters: dict[str, Any] = {}
        self._engine: DrugGNNEngine | None = None

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        return {
            "hidden_dim": 64,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "epochs": 2,
            "batch_size": 8,
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
        _ = hyperparameters, model_name
        return DrugGNNEngine().load_cell_line_features(data_path, dataset_name), {}

    @classmethod
    def load_dataset_drug_features(
        cls,
        data_path: str,
        dataset_name: str,
        *,
        hyperparameters: dict[str, Any] | None = None,
        model_name: str | None = None,
    ) -> FeatureDataset | None:
        _ = hyperparameters, model_name
        return DrugGNNEngine().load_drug_features(data_path, dataset_name)

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {
            "hidden_dim": {"type": "int", "low": 16, "high": 128, "default": 64},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5, "default": 0.2},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True, "default": 1e-3},
            "epochs": {"type": "int", "low": 1, "high": 10, "default": 2},
            "batch_size": {"type": "int", "low": 4, "high": 32, "default": 8},
        }

    def build(self, hyperparameters: dict[str, Any], input_dims: dict[str, Any]) -> None:
        _ = input_dims
        self._hyperparameters = {**self.get_default_hyperparameters(), **hyperparameters}
        self._engine = None

    def fit(self, batch: ModelInputBatch) -> None:
        cell_line_input = batch.cell_line_input
        drug_input = batch.drug_input
        if batch.response is None:
            msg = "DrugGNN requires response data"
            raise RuntimeError(msg)
        if cell_line_input is None or drug_input is None:
            msg = "DrugGNN requires cell_line_input and drug_input"
            raise RuntimeError(msg)
        output = DrugResponseDataset(
            response=batch.response,
            cell_line_ids=batch.cell_line_ids,
            drug_ids=batch.drug_ids,
        )
        engine = DrugGNNEngine()
        engine.configure(dict(self._hyperparameters))
        engine.train(
            output,
            cell_line_input,
            drug_input,
            output_earlystopping=batch.early_stopping_response,
            model_checkpoint_dir=batch.training_context.checkpoint_dir,
        )
        self._engine = engine

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        cell_line_input = batch.cell_line_input
        drug_input = batch.drug_input
        if self._engine is None or cell_line_input is None or drug_input is None:
            return np.full(batch.n_pairs, np.nan, dtype=np.float64)
        return self._engine.predict(
            batch.cell_line_ids,
            batch.drug_ids,
            cell_line_input,
            drug_input,
        )

    def is_fitted(self) -> bool:
        return self._engine is not None and self._engine.model is not None

    def get_state(self) -> dict[str, object]:
        if self._engine is None:
            return {}
        buffer = io.BytesIO()
        joblib.dump(self._engine, buffer)
        return {
            "hyperparameters": dict(self._hyperparameters),
            "engine": buffer.getvalue(),
        }

    def set_state(self, state: dict[str, object]) -> None:
        engine_blob = state.get("engine")
        if not isinstance(engine_blob, (bytes, bytearray)):
            msg = "DrugGNNPredictor state requires an engine byte blob"
            raise PredictorStateError(msg)
        try:
            loaded = joblib.load(io.BytesIO(engine_blob))
        except Exception as exc:
            msg = "DrugGNNPredictor engine blob could not be deserialized"
            raise PredictorStateError(msg) from exc
        if not isinstance(loaded, DrugGNNEngine):
            msg = "DrugGNNPredictor engine blob did not deserialize to DrugGNN"
            raise PredictorStateError(msg)
        if loaded.model is None:
            msg = "DrugGNNPredictor engine blob is missing a trained model"
            raise PredictorStateError(msg)
        hyperparameters = state.get("hyperparameters")
        if not isinstance(hyperparameters, dict):
            msg = "DrugGNNPredictor state is missing hyperparameters"
            raise PredictorStateError(msg)
        self._engine = loaded
        self._hyperparameters = dict(hyperparameters)
