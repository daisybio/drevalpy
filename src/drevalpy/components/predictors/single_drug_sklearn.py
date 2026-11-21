"""Per-drug scikit-learn estimators on dense sample features."""

from __future__ import annotations

from typing import Any

import numpy as np

from drevalpy.components.predictors.abstract.base import get_response
from drevalpy.components.predictors.abstract.single_drug import SingleDrugPredictor
from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor
from drevalpy.types.data.predictor_input import PredictorInput


class SingleDrugSklearnPredictor(SingleDrugPredictor, SklearnTabularPredictor):
    """Fit one ``_make_estimator()`` estimator per drug on the sample matrix of its pairs."""

    def _fit_drug(self, drug_id: str, data: PredictorInput) -> Any:
        """Fit the estimator of one drug.

        :param drug_id: Drug whose pairs *data* holds; the estimator does not depend on it.
        :param data: The drug's training pairs.
        :returns: The fitted estimator.
        """
        estimator = self._make_estimator()
        estimator.fit(_sample_matrix(data), get_response(data).ravel())
        return estimator

    def _predict_drug(self, drug_id: str, model: Any, data: PredictorInput) -> np.ndarray:
        """Score the pairs of one drug with its estimator.

        :param drug_id: Drug whose pairs *data* holds.
        :param model: The drug's fitted estimator.
        :param data: The drug's pairs to score.
        :returns: One prediction per pair.
        """
        return model.predict(_sample_matrix(data))


def _sample_matrix(data: PredictorInput) -> np.ndarray:
    """Expand the entity-level sample features to one row per pair.

    :param data: Featurized batch.
    :returns: Pair-level sample feature matrix, with no columns when the batch has no sample features.
    """
    if data.sample.matrix.size == 0:
        return np.empty((len(data), 0), dtype=np.float32)
    return data.sample.matrix[data.sample_pair_idx]
