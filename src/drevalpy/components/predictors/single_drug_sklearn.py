"""Per-drug scikit-learn estimators on dense sample features."""

from __future__ import annotations

from typing import Any

import numpy as np

from drevalpy.components.predictors.abstract.base import fit_response
from drevalpy.components.predictors.abstract.single_drug import SingleDrugPredictor
from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


class SingleDrugSklearnPredictor(SingleDrugPredictor, SklearnTabularPredictor):
    """Fit one ``_make_estimator()`` estimator per drug on the sample matrix of its pairs."""

    def _fit_drug(self, drug_id: str, batch: ModelInputBatch) -> Any:
        """Fit the estimator of one drug.

        :param drug_id: Drug whose pairs *batch* holds; the estimator does not depend on it.
        :param batch: The drug's training pairs.
        :returns: The fitted estimator.
        """
        estimator = self._make_estimator()
        estimator.fit(_sample_matrix(batch), fit_response(batch).ravel())
        return estimator

    def _predict_drug(self, drug_id: str, model: Any, batch: ModelInputBatch) -> np.ndarray:
        """Score the pairs of one drug with its estimator.

        :param drug_id: Drug whose pairs *batch* holds.
        :param model: The drug's fitted estimator.
        :param batch: The drug's pairs to score.
        :returns: One prediction per pair.
        """
        return model.predict(_sample_matrix(batch))


def _sample_matrix(batch: ModelInputBatch) -> np.ndarray:
    """Expand the entity-level sample features to one row per pair.

    :param batch: Featurized batch.
    :returns: Pair-level sample feature matrix, with no columns when the batch has no sample features.
    """
    if batch.sample.matrix.size == 0:
        return np.empty((batch.n_pairs, 0), dtype=np.float32)
    return batch.sample.matrix[batch.sample_pair_idx]
