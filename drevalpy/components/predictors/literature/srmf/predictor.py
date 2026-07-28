"""SRMF block literature predictor."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.literature._algorithm_lifecycle import (
    predict_with_algorithm,
    train_fitted_algorithm,
)
from drevalpy.components.predictors.literature._block_inputs import materialize_block_inputs
from drevalpy.components.predictors.literature._metadata import SRMF_REFERENCE
from drevalpy.components.predictors.literature._preload import (
    load_dataset_cell_line_features,
    load_dataset_drug_features,
)
from drevalpy.components.predictors.literature._torch_state import load_object_mapping, save_object_mapping
from drevalpy.components.predictors.literature.srmf.algorithm import SRMF
from drevalpy.components.predictors.literature.srmf.state import apply_state, export_state
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.predictors.structured import BlockPredictor
from drevalpy.components.registry import register_predictor
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.models.config import PredictionMode


@register_predictor(
    "srmf",
    description="SRMF matrix factorization model.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=SRMF_REFERENCE,
)
class SRMFPredictor(BlockPredictor):
    """Registered SRMF predictor."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("gene_expression",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("fingerprints",)
    requires_drug_featurizer: ClassVar[bool] = True
    supports_early_stopping: ClassVar[bool] = False
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        super().__init__(hyperparameters)
        self._algorithm: SRMF | None = None
        self._engine_preload_state: dict[str, Any] = {}

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        return dict(SRMF.get_default_hyperparameters())

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        space = getattr(SRMF, "get_hyperparameter_space", None)
        if callable(space):
            return dict(space())
        return {}

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
        return load_dataset_cell_line_features(SRMF, data_path, dataset_name, hyperparameters=hyperparameters)

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
        return load_dataset_drug_features(SRMF, data_path, dataset_name, hyperparameters=hyperparameters)

    def set_engine_preload_state(self, state: dict[str, Any]) -> None:
        self._engine_preload_state = dict(state)

    def fit(self, batch: ModelInputBatch) -> None:
        cell_lines, drugs = materialize_block_inputs(
            self,
            batch,
            required_cell_line_blocks=self.required_cell_line_blocks,
            required_drug_blocks=self.required_drug_blocks,
            requires_drug_featurizer=self.requires_drug_featurizer,
        )
        self._algorithm = train_fitted_algorithm(
            SRMF,
            dict(self._hyperparameters),
            self._engine_preload_state,
            batch,
            cell_lines,
            drugs,
        )

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        cell_lines, drugs = materialize_block_inputs(
            self,
            batch,
            required_cell_line_blocks=self.required_cell_line_blocks,
            required_drug_blocks=self.required_drug_blocks,
            requires_drug_featurizer=self.requires_drug_featurizer,
        )
        return predict_with_algorithm(self._algorithm, batch, cell_lines, drugs)

    def is_fitted(self) -> bool:
        return self._algorithm is not None and not self._algorithm.best_u.empty

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
        self._hyperparameters = dict(hyperparameters)
        self._algorithm = apply_state(payload)
