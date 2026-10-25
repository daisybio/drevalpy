"""Apply a fitted model to every observed pair of a dataset."""

from __future__ import annotations

import numpy as np

from drevalpy.evaluation._scoring import run_metrics
from drevalpy.models.drp_model import DRPModel
from drevalpy.types import SplitMask
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.results.run import RunResult


def predict_observed(model: DRPModel, dataset: Dataset) -> RunResult:
    """Predict every pair of *dataset* that has an observed response.

    The dataset need not be the one the model was trained on. Predictions are in the units of the
    response, since the model inverts its response transformation. A pair the model has no
    features for gets ``NaN``.

    :param model: Fitted model, such as one from :func:`drevalpy.train` or ``DRPModel.load``.
    :param dataset: Dataset to apply the model to.
    :returns: The predictions with the observed responses and ids of the same pairs, in the layout of a run file.
        ``metrics`` score the model on those pairs, so for the training dataset they are in-sample.
    """
    scope = SplitMask(np.isfinite(dataset.response_matrix))
    observed = ResponseBatch.from_dataset(dataset, scope)
    predictions = model.predict(dataset, scope)
    return RunResult(
        model_name=model.name,
        dataset_name=dataset.name,
        fold_index=0,
        predictions=predictions,
        ground_truth=observed.response,
        sample_ids=observed.sample_ids,
        drug_ids=observed.drug_ids,
        best_hyperparameters=model.hyperparameters,
        metrics=run_metrics(predictions, observed.response),
        randomization=dataset.randomization,
    )
