"""Shared per-drug routing for scikit-learn matrix predictors."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.single_drug_routing import iter_drug_masks, routing_keys, run_per_drug
from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.enums.model_scope import ModelScope


class SingleDrugSklearnPredictor(SklearnTabularPredictor):
    """Fit one estimator per drug on sample features, routing pairs by ``batch.drug_ids``."""

    drug_inputs: ClassVar[Inputs] = Inputs.none()
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
            if estimator is None:
                continue
            mask = keys == drug_id
            predictions[mask] = np.asarray(estimator.predict(x[mask]), dtype=np.float64)
        return predictions
