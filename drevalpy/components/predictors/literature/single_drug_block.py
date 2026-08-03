"""Shared per-drug routing for literature block predictors."""

from __future__ import annotations

import hashlib
from abc import abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.literature._algorithm_lifecycle import (
    predict_with_algorithm,
    train_fitted_algorithm,
)
from drevalpy.components.predictors.literature._block_inputs import materialize_block_inputs
from drevalpy.components.predictors.literature._torch_state import load_object_mapping, save_object_mapping
from drevalpy.components.predictors.literature._training_helpers import LiteratureTrainingMixin
from drevalpy.components.predictors.single_drug_routing import (
    iter_drug_masks,
    require_known_training_keys,
    routing_keys,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.predictors.structured import BlockPredictor
from drevalpy.components.training_context import TrainingContext
from drevalpy.models.config import ModelScope


def _checkpoint_dir_for_drug(base_dir: str, drug_id: str) -> str:
    digest = hashlib.sha256(drug_id.encode()).hexdigest()[:16]
    return str(Path(base_dir) / f"drug_{digest}")


class SingleDrugBlockPredictor(BlockPredictor):
    """Fit one literature algorithm per drug using identity routing."""

    supported_scopes: ClassVar[frozenset[ModelScope]] = frozenset({ModelScope.SINGLE_DRUG})
    routing_drug_featurizer: ClassVar[str] = "identity"
    requires_drug_featurizer: ClassVar[bool] = True

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        super().__init__(hyperparameters)
        self._algorithms: dict[str, LiteratureTrainingMixin] = {}
        self._legacy_algorithm: LiteratureTrainingMixin | None = None
        self._engine_preload_state: dict[str, Any] = {}

    @property
    @abstractmethod
    def _algorithm_cls(self) -> type[LiteratureTrainingMixin]:
        """Algorithm class trained for each routed drug."""

    @abstractmethod
    def _export_algorithm_state(self, algorithm: LiteratureTrainingMixin) -> dict[str, Any]:
        """Serialize one fitted algorithm."""

    @abstractmethod
    def _apply_algorithm_state(self, payload: dict[str, Any]) -> LiteratureTrainingMixin:
        """Restore one fitted algorithm from a serialized payload."""

    def fit(self, batch: ModelInputBatch) -> None:
        keys = routing_keys(batch)
        require_known_training_keys(keys)
        self._algorithms = {}
        self._legacy_algorithm = None
        for drug_id, mask in iter_drug_masks(batch):
            context = TrainingContext(
                checkpoint_dir=_checkpoint_dir_for_drug(batch.training_context.checkpoint_dir, drug_id),
                logging_metadata=dict(batch.training_context.logging_metadata),
            )
            sub = replace(batch.subset_pairs(mask), training_context=context)
            cell_lines, _ = materialize_block_inputs(
                self,
                sub,
                required_cell_line_blocks=self.required_cell_line_blocks,
                required_drug_blocks=self.required_drug_blocks,
                requires_drug_featurizer=True,
            )
            self._algorithms[drug_id] = train_fitted_algorithm(
                self._algorithm_cls,
                dict(self._hyperparameters),
                self._engine_preload_state,
                sub,
                cell_lines,
                None,
            )

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        keys = routing_keys(batch)
        predictions = np.full(batch.n_pairs, np.nan, dtype=np.float64)
        for drug_id in np.unique(keys):
            if drug_id == "":
                continue
            algorithm = self._resolve_algorithm(str(drug_id), keys)
            if algorithm is None:
                continue
            mask = keys == drug_id
            sub = batch.subset_pairs(mask)
            cell_lines, _ = materialize_block_inputs(
                self,
                sub,
                required_cell_line_blocks=self.required_cell_line_blocks,
                required_drug_blocks=self.required_drug_blocks,
                requires_drug_featurizer=True,
            )
            routed = predict_with_algorithm(algorithm, sub, cell_lines, None)
            predictions[mask] = np.asarray(routed, dtype=np.float64).ravel()
        return predictions

    def _resolve_algorithm(self, drug_id: str, keys: np.ndarray) -> LiteratureTrainingMixin | None:
        algorithm = self._algorithms.get(drug_id)
        if algorithm is not None:
            return algorithm
        if self._legacy_algorithm is None:
            return None
        unique_drugs = {str(value) for value in np.unique(keys) if value != ""}
        if unique_drugs != {drug_id}:
            msg = (
                f"{self.__class__.__name__} legacy state supports only single-drug batches; "
                f"requested drugs={sorted(unique_drugs)}"
            )
            raise PredictorStateError(msg)
        return self._legacy_algorithm

    def is_fitted(self) -> bool:
        return bool(self._algorithms) or self._legacy_algorithm is not None

    def get_state(self) -> dict[str, object]:
        if not self._algorithms:
            return {}
        algorithms = {
            drug_id: save_object_mapping(self._export_algorithm_state(algorithm))
            for drug_id, algorithm in self._algorithms.items()
        }
        return {
            "algorithms": algorithms,
            "predictor_hyperparameters": dict(self._hyperparameters),
        }

    def _load_algorithm_blob(self, drug_id: str, blob: object) -> LiteratureTrainingMixin:
        if not isinstance(blob, (bytes, bytearray)):
            msg = f"{self.__class__.__name__} algorithm payload for {drug_id!r} must be bytes"
            raise PredictorStateError(msg)
        try:
            payload = load_object_mapping(bytes(blob))
        except Exception as exc:
            msg = f"{self.__class__.__name__} algorithm payload for {drug_id!r} could not be deserialized"
            raise PredictorStateError(msg) from exc
        return self._apply_algorithm_state(payload)

    def _set_state_from_algorithms(self, state: dict[str, object], algorithms_blob: dict[object, object]) -> None:
        hyperparameters = state.get("predictor_hyperparameters")
        if not isinstance(hyperparameters, dict):
            msg = f"{self.__class__.__name__} state is missing predictor_hyperparameters"
            raise PredictorStateError(msg)
        self._hyperparameters = dict(hyperparameters)
        self._algorithms = {
            str(drug_id): self._load_algorithm_blob(str(drug_id), blob) for drug_id, blob in algorithms_blob.items()
        }
        self._legacy_algorithm = None

    def _set_state_from_legacy_payload(self, blob: bytes | bytearray) -> None:
        try:
            payload = load_object_mapping(bytes(blob))
        except Exception as exc:
            msg = f"{self.__class__.__name__} payload could not be deserialized"
            raise PredictorStateError(msg) from exc
        hyperparameters = payload.get("predictor_hyperparameters")
        if not isinstance(hyperparameters, dict):
            msg = f"{self.__class__.__name__} payload is missing predictor_hyperparameters"
            raise PredictorStateError(msg)
        self._hyperparameters = dict(hyperparameters)
        algorithm_payload = {key: value for key, value in payload.items() if key != "predictor_hyperparameters"}
        self._algorithms = {}
        self._legacy_algorithm = self._apply_algorithm_state(algorithm_payload)

    def set_state(self, state: dict[str, object]) -> None:
        algorithms_blob = state.get("algorithms")
        if isinstance(algorithms_blob, dict):
            self._set_state_from_algorithms(state, algorithms_blob)
            return

        blob = state.get("payload")
        if not isinstance(blob, (bytes, bytearray)):
            msg = f"{self.__class__.__name__} state requires algorithms or a legacy payload"
            raise PredictorStateError(msg)
        self._set_state_from_legacy_payload(blob)
