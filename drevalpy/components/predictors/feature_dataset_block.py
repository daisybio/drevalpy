"""Deprecated adapter for FeatureDataset-protocol predictor cores."""

from __future__ import annotations

from abc import abstractmethod
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
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.datasets.dataset import FeatureDataset

_DEPRECATION_MESSAGE = (
    "FeatureDatasetBlockPredictor is a legacy adapter for FeatureDataset-protocol cores. "
    "New predictors should subclass FeatureFreePredictor, MatrixPredictor, or BlockPredictor "
    "and consume ModelInputBatch directly (see docs/python/custom_models.rst)."
)


@deprecated(_DEPRECATION_MESSAGE, category=None)
class FeatureDatasetBlockPredictor(BlockPredictor):
    """Deprecated adapter that materializes ``FeatureDataset`` views for a core.

    **Deprecated.** Prefer modern predictors that consume ``ModelInputBatch``
    directly:

    - ``FeatureFreePredictor`` — response / ids only
    - ``MatrixPredictor`` — flattened dense pair features
    - ``BlockPredictor`` — named typed blocks on the batch
    - ``SingleDrugSklearnPredictor`` — per-drug dense estimators

    ``FeatureDataset`` remains valid as DRPModel / featurizer I/O. This class
    only exists for literature (and similar) cores that still call
    ``train`` / ``predict`` with ``FeatureDataset``.
    """

    validate_drug_graphs: ClassVar[bool] = False

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize the predictor.

        :param hyperparameters: Optional overrides for algorithm defaults.
        """
        super().__init__(hyperparameters)
        self._algorithm: LiteratureTrainingMixin | None = None
        self._engine_preload_state: dict[str, Any] = {}

    @property
    @abstractmethod
    def _algorithm_cls(self) -> type[LiteratureTrainingMixin]:
        """FeatureDataset-protocol core class trained for all pairs."""

    @abstractmethod
    def _export_algorithm_state(self, algorithm: LiteratureTrainingMixin) -> dict[str, Any]:
        """Serialize one fitted core.

        :param algorithm: Fitted core instance.

        :returns: Mapping suitable for binary payload serialization.
        """

    @abstractmethod
    def _apply_algorithm_state(self, payload: dict[str, Any]) -> LiteratureTrainingMixin:
        """Restore one fitted core from a serialized payload.

        :param payload: Serialized core state.

        :returns: Restored core instance.
        """

    def set_engine_preload_state(self, state: dict[str, Any]) -> None:
        """Store engine preload attributes applied before algorithm training.

        :param state: Attribute mapping copied onto the algorithm before fit.
        """
        self._engine_preload_state = dict(state)

    def _materialize_inputs(self, batch: ModelInputBatch) -> tuple[FeatureDataset, FeatureDataset | None]:
        """Build FeatureDataset views from batch blocks.

        :param batch: Featurized pairs with named blocks.

        :returns: Cell-line dataset and optional drug dataset.
        """
        return materialize_block_inputs(
            self,
            batch,
            required_cell_line_blocks=self.required_cell_line_blocks,
            required_drug_blocks=self.required_drug_blocks,
            validate_drug_graphs=self.validate_drug_graphs,
        )

    def _is_algorithm_fitted(self, algorithm: LiteratureTrainingMixin | None) -> bool:
        """Return whether *algorithm* counts as fitted.

        :param algorithm: Current core instance, or ``None``.

        :returns: ``True`` when a fitted core is loaded.
        """
        return algorithm is not None

    def _validate_restored_payload(self, payload: dict[str, Any]) -> None:
        """Validate a deserialized payload before applying algorithm state.

        :param payload: Mapping loaded from the state blob.
        """

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train the underlying algorithm on featurized pairs.

        :param batch: Training batch with responses and feature blocks.
        """
        cell_lines, drugs = self._materialize_inputs(batch)
        self._algorithm = train_fitted_algorithm(
            self._algorithm_cls,
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
        cell_lines, drugs = self._materialize_inputs(batch)
        return predict_with_algorithm(self._algorithm, batch, cell_lines, drugs)

    def is_fitted(self) -> bool:
        """Report whether a trained algorithm is loaded.

        :returns: ``True`` when the algorithm has been fit or restored.
        """
        return self._is_algorithm_fitted(self._algorithm)

    def get_state(self) -> dict[str, object]:
        """Serialize fitted predictor state.

        :returns: Mapping with a binary ``payload`` blob when fitted, else empty.
        """
        if self._algorithm is None:
            return {}
        payload = self._export_algorithm_state(self._algorithm)
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
        self._validate_restored_payload(payload)
        self._hyperparameters = dict(hyperparameters)
        self._algorithm = self._apply_algorithm_state(payload)
