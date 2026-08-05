"""DrugGNN block literature predictor."""

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
from drevalpy.components.predictors.literature._metadata import DRUGGNN_REFERENCE
from drevalpy.components.predictors.literature._torch_state import load_object_mapping, save_object_mapping
from drevalpy.components.predictors.literature.druggnn.algorithm import DrugGNN
from drevalpy.components.predictors.literature.druggnn.state import apply_state, export_state
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.predictors.structured import BlockPredictor
from drevalpy.components.registry import register_predictor
from drevalpy.models.config import PredictionMode


@register_predictor(
    "drugGNN",
    description="DrugGNN: GCN on molecular graphs with dense cell-line features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.GRAPH,
    reference=DRUGGNN_REFERENCE,
)
class DrugGNNPredictor(BlockPredictor):
    """Registered DrugGNN predictor."""

    supports_early_stopping: ClassVar[bool] = True
    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("gene_expression",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("drug_graph",)
    required_cell_line_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
    )
    required_drug_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("drug_graph", FeatureFormat.GRAPH),)
    requires_drug_featurizer: ClassVar[bool] = True
    validate_drug_graphs: ClassVar[bool] = True
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize the predictor.

        :param hyperparameters: Optional overrides for algorithm defaults.
        """
        super().__init__(hyperparameters)
        self._algorithm: DrugGNN | None = None
        self._engine_preload_state: dict[str, Any] = {}

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameters from the algorithm class.

        :returns: Default hyperparameter mapping.
        """
        defaults = dict(DrugGNN.get_default_hyperparameters())
        defaults.update({"epochs": 2, "batch_size": 8})
        return defaults

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space when exposed by the algorithm.

        :returns: Ray Tune-style hyperparameter specs.
        """
        return {
            "hidden_dim": {"type": "int", "low": 16, "high": 128, "default": 64},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5, "default": 0.2},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True, "default": 1e-3},
            "epochs": {"type": "int", "low": 1, "high": 10, "default": 2},
            "batch_size": {"type": "int", "low": 4, "high": 32, "default": 8},
        }

    def _materialized_inputs(self, batch: ModelInputBatch):
        cell_lines, drugs = materialize_block_inputs(
            self,
            batch,
            required_cell_line_blocks=self.required_cell_line_blocks,
            required_drug_blocks=self.required_drug_blocks,
            requires_drug_featurizer=self.requires_drug_featurizer,
            validate_drug_graphs=self.validate_drug_graphs,
        )
        if drugs is None:
            raise ValueError("DrugGNN requires drug graph blocks")
        return cell_lines, drugs

    def fit(self, batch: ModelInputBatch) -> None:
        """Train the underlying algorithm on featurized pairs.

        :param batch: Training batch with responses and feature blocks.
        """
        cell_lines, drugs = self._materialized_inputs(batch)
        self._algorithm = train_fitted_algorithm(
            DrugGNN,
            dict(self._hyperparameters),
            self._engine_preload_state,
            batch,
            cell_lines,
            drugs,
        )

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict responses for pairs in the batch.

        :param batch: Featurized pairs to score.

        :returns: One predicted response per pair.
        """
        cell_lines, drugs = self._materialized_inputs(batch)
        return predict_with_algorithm(self._algorithm, batch, cell_lines, drugs)

    def is_fitted(self) -> bool:
        """Report whether a trained algorithm is loaded.

        :returns: ``True`` when the algorithm has been fit or restored.
        """
        return self._algorithm is not None and self._algorithm.model is not None

    def get_state(self) -> dict[str, object]:
        """Serialize fitted predictor state.

        :returns: Mapping with a binary ``payload`` blob when fitted, else empty.
        """
        if self._algorithm is None:
            return {}
        payload = export_state(self._algorithm)
        payload["predictor_hyperparameters"] = dict(self._hyperparameters)
        return {"payload": save_object_mapping(payload)}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore a predictor from ``get_state`` output.

        :param state: Serialized state containing a ``payload`` byte blob.

        :raises PredictorStateError: If the payload is missing or invalid.
        """
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
