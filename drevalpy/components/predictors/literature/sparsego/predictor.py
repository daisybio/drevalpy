"""sparsego literature predictor."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import BlockSpec
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.literature._algorithm_lifecycle import (
    predict_with_algorithm,
    train_fitted_algorithm,
)
from drevalpy.components.predictors.literature._block_inputs import materialize_block_inputs
from drevalpy.components.predictors.literature._metadata import SPARSEGO_REFERENCE
from drevalpy.components.predictors.literature._torch_state import load_object_mapping, save_object_mapping
from drevalpy.components.predictors.literature.sparsego.algorithm import SparseGOModel
from drevalpy.components.predictors.literature.sparsego.state import apply_state, export_state
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.predictors.structured import BlockPredictor
from drevalpy.components.registry import register_predictor
from drevalpy.models.config import PredictionMode


@register_predictor(
    "sparsego",
    description="SparseGO GO-structured visible neural network.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=SPARSEGO_REFERENCE,
)
class SparseGOPredictor(BlockPredictor):
    """Registered sparsego predictor."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ()
    required_cell_line_block_alternatives: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX, metadata=True),
        BlockSpec("mutations", FeatureFormat.NUMERIC_MATRIX, metadata=True),
    )
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("fingerprints",)
    required_drug_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),
    )
    requires_drug_featurizer: ClassVar[bool] = True
    validate_drug_graphs: ClassVar[bool] = False
    supports_early_stopping: ClassVar[bool] = False
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        super().__init__(hyperparameters)
        self._algorithm: SparseGOModel | None = None
        self._engine_preload_state: dict[str, Any] = {}

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        return dict(SparseGOModel.get_default_hyperparameters())

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        space = getattr(SparseGOModel, "get_hyperparameter_space", None)
        if callable(space):
            return dict(space())
        return {}

    def _materialized_inputs(self, batch: ModelInputBatch):
        active = [
            spec.name for spec in self.required_cell_line_block_alternatives if spec.name in batch.cell_line_blocks
        ]
        if len(active) != 1:
            raise ValueError(
                "SparseGOPredictor requires exactly one cell-line block from " "['gene_expression', 'mutations']"
            )
        block = batch.cell_line_blocks[active[0]]
        if block.metadata is None:
            raise ValueError("SparseGOPredictor requires ontology metadata on its active cell-line block")
        self._engine_preload_state = dict(block.metadata)
        return materialize_block_inputs(
            self,
            batch,
            required_cell_line_blocks=(active[0],),
            required_drug_blocks=self.required_drug_blocks,
            requires_drug_featurizer=self.requires_drug_featurizer,
        )

    def fit(self, batch: ModelInputBatch) -> None:
        cell_lines, drugs = self._materialized_inputs(batch)
        self._algorithm = train_fitted_algorithm(
            SparseGOModel,
            dict(self._hyperparameters),
            self._engine_preload_state,
            batch,
            cell_lines,
            drugs,
        )

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        cell_lines, drugs = self._materialized_inputs(batch)
        return predict_with_algorithm(self._algorithm, batch, cell_lines, drugs)

    def is_fitted(self) -> bool:
        return self._algorithm is not None

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
