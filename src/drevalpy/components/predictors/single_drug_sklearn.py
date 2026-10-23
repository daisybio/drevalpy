"""Shared per-drug routing for scikit-learn matrix predictors."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.predictors._state_helpers import state_mapping
from drevalpy.components.predictors.single_drug_routing import (
    iter_drug_masks,
    require_known_training_keys,
    routing_keys,
    run_per_drug,
)
from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.enums.model_scope import ModelScope


class SingleDrugSklearnPredictor(SklearnTabularPredictor):
    """Fit one estimator per drug, using drug identity only for routing."""

    scope: ClassVar[ModelScope] = ModelScope.SINGLE_DRUG

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize instance state.

        :param hyperparameters: Optional hyperparameter overrides, passed to the base class.
        """
        super().__init__(hyperparameters)
        self._estimators: dict[str, Any] = {}

    @staticmethod
    def _sample_matrix(batch: ModelInputBatch) -> np.ndarray:
        """Expand deduplicated sample features to one row per pair.

        Features are stored entity-level (one row per unique sample).
        ``sample_pair_idx`` maps each pair to its sample row, repeating
        rows when multiple pairs share the same sample.

        :param batch: Featurized batch.
        :returns: Pair-level sample feature matrix.
        """
        if batch.sample.matrix.size == 0:
            return np.empty((batch.n_pairs, 0), dtype=np.float32)
        return batch.sample.matrix[batch.sample_pair_idx]

    def _fit(self, batch: ModelInputBatch) -> None:
        """Fit on training data.

        :param batch: Training batch; one estimator is fitted on the pairs of each known drug, in
            ``batch.training_context.n_jobs`` worker processes.
        :raises RuntimeError: If ``batch.response`` is None.
        """
        x = self._sample_matrix(batch)
        if batch.response is None:
            raise RuntimeError("batch.response is required for fit")
        y = batch.response.ravel()
        keys = routing_keys(batch)
        require_known_training_keys(keys)

        self._estimators = {}
        tasks = ((drug_id, (x[mask], y[mask])) for drug_id, mask in iter_drug_masks(batch))
        self._estimators = run_per_drug(self._fit_estimator, tasks, batch.training_context.n_jobs)

    def _fit_estimator(self, _drug_id: str, data: tuple[np.ndarray, np.ndarray]) -> Any:
        """Fit the estimator of one drug.

        :param _drug_id: Drug identifier; the estimator does not depend on it.
        :param data: Sample features and responses of the drug's pairs.
        :returns: The fitted estimator.
        """
        estimator = self._make_estimator()
        estimator.fit(*data)
        return estimator

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict responses for each pair in the batch.

        :param batch: Pairs to score.
        :returns: One prediction per pair; NaN for pairs of drugs without a fitted estimator.
        """
        x = self._sample_matrix(batch)
        keys = routing_keys(batch)
        predictions = np.full(batch.n_pairs, np.nan, dtype=np.float64)
        for drug_id in np.unique(keys):
            estimator = self._estimators.get(str(drug_id))
            if drug_id == "" or estimator is None:
                continue
            mask = keys == drug_id
            predictions[mask] = np.asarray(estimator.predict(x[mask]), dtype=np.float64)
        return predictions

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state.

        :returns: The per-drug estimators plus the shared hyperparameter and mode entries.
        """
        return {"estimators": dict(self._estimators), **self._shared_state()}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: Mapping produced by ``get_state``.
        :raises PredictorStateError: If the per-drug estimators or the hyperparameters are missing.
        """
        estimators = state_mapping(state, "estimators")
        if not estimators:
            msg = f"{self.__class__.__name__} state is missing fitted per-drug estimators"
            raise PredictorStateError(msg)
        shared = self._validated_shared_state(state)
        self._estimators = {str(key): value for key, value in estimators.items()}
        self._apply_shared_state(shared)

    def is_fitted(self) -> bool:
        """Return whether the component has been fit.

        :returns: True once at least one per-drug estimator exists.
        """
        return bool(self._estimators)
