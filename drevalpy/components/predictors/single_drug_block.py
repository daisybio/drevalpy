"""Deprecated per-drug FeatureDataset-protocol block predictor routing."""

from __future__ import annotations

import hashlib
from abc import abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from typing_extensions import deprecated

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.block import BlockPredictor
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
from drevalpy.components.training_context import TrainingContext
from drevalpy.models.config import ModelScope

_DEPRECATION_MESSAGE = (
    "SingleDrugBlockPredictor is a legacy adapter for FeatureDataset-protocol cores. "
    "New predictors should subclass FeatureFreePredictor, MatrixPredictor, or BlockPredictor "
    "and consume ModelInputBatch directly (see docs/python/custom_models.rst). "
    "For per-drug dense estimators prefer SingleDrugSklearnPredictor."
)


def _checkpoint_dir_for_drug(base_dir: Path, drug_id: str) -> Path:
    digest = hashlib.sha256(drug_id.encode()).hexdigest()[:16]
    return base_dir / f"drug_{digest}"


@deprecated(_DEPRECATION_MESSAGE, category=None)
class SingleDrugBlockPredictor(BlockPredictor):
    """Deprecated adapter: one FeatureDataset-protocol core per drug.

    **Deprecated.** Prefer modern predictors that consume ``ModelInputBatch``
    directly:

    - ``FeatureFreePredictor`` — response / ids only
    - ``MatrixPredictor`` — flattened dense pair features
    - ``BlockPredictor`` — named typed blocks on the batch
    - ``SingleDrugSklearnPredictor`` — per-drug dense estimators

    ``FeatureDataset`` remains valid as DRPModel / featurizer I/O. This class
    only exists for literature (and similar) single-drug cores that still call
    ``train`` / ``predict`` with ``FeatureDataset``.
    """

    scope: ClassVar[ModelScope] = ModelScope.SINGLE_DRUG

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize the predictor.

        :param hyperparameters: Optional overrides for algorithm defaults.
        """
        super().__init__(hyperparameters)
        self._algorithms: dict[str, LiteratureTrainingMixin] = {}
        self._engine_preload_state: dict[str, Any] = {}

    @property
    @abstractmethod
    def _algorithm_cls(self) -> type[LiteratureTrainingMixin]:
        """Algorithm class trained for each routed drug."""

    @abstractmethod
    def _export_algorithm_state(self, algorithm: LiteratureTrainingMixin) -> dict[str, Any]:
        """Serialize one fitted algorithm.

        :param algorithm: Fitted per-drug algorithm instance.

        :returns: JSON-serializable algorithm state.
        """

    @abstractmethod
    def _apply_algorithm_state(self, payload: dict[str, Any]) -> LiteratureTrainingMixin:
        """Restore one fitted algorithm from a serialized payload.

        :param payload: Serialized algorithm state for one drug.

        :returns: Restored algorithm instance.
        """

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train the underlying algorithm on featurized pairs.

        :param batch: Training batch with responses and feature blocks.
        """
        keys = routing_keys(batch)
        require_known_training_keys(keys)
        self._algorithms = {}
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
        """Predict responses for pairs in the batch.

        :param batch: Featurized pairs to score.

        :returns: One predicted response per pair.
        """
        keys = routing_keys(batch)
        predictions = np.full(batch.n_pairs, np.nan, dtype=np.float64)
        for drug_id in np.unique(keys):
            if drug_id == "":
                continue
            algorithm = self._algorithms.get(str(drug_id))
            if algorithm is None:
                continue
            mask = keys == drug_id
            sub = batch.subset_pairs(mask)
            cell_lines, _ = materialize_block_inputs(
                self,
                sub,
                required_cell_line_blocks=self.required_cell_line_blocks,
                required_drug_blocks=self.required_drug_blocks,
            )
            routed = predict_with_algorithm(algorithm, sub, cell_lines, None)
            predictions[mask] = np.asarray(routed, dtype=np.float64).ravel()
        return predictions

    def is_fitted(self) -> bool:
        """Report whether a trained algorithm is loaded.

        :returns: ``True`` when the algorithm has been fit or restored.
        """
        return bool(self._algorithms)

    def get_state(self) -> dict[str, object]:
        """Serialize fitted predictor state.

        :returns: Mapping with per-drug algorithm blobs when fitted, else empty.
        """
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

    def set_state(self, state: dict[str, object]) -> None:
        """Restore a predictor from ``get_state`` output.

        :param state: Serialized state with per-drug algorithm blobs.

        :raises PredictorStateError: If the state is missing or invalid.
        """
        algorithms_blob = state.get("algorithms")
        if not isinstance(algorithms_blob, dict):
            msg = f"{self.__class__.__name__} state requires an 'algorithms' mapping"
            raise PredictorStateError(msg)
        hyperparameters = state.get("predictor_hyperparameters")
        if not isinstance(hyperparameters, dict):
            msg = f"{self.__class__.__name__} state is missing predictor_hyperparameters"
            raise PredictorStateError(msg)
        self._hyperparameters = dict(hyperparameters)
        self._algorithms = {
            str(drug_id): self._load_algorithm_blob(str(drug_id), blob) for drug_id, blob in algorithms_blob.items()
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
