"""Train a literature algorithm instance from a prepared input batch."""

from __future__ import annotations

from typing import Any, TypeVar, cast

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.literature._training_helpers import LiteratureTrainingMixin
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

TAlgorithm = TypeVar("TAlgorithm", bound=LiteratureTrainingMixin)


def train_fitted_algorithm(
    algorithm_cls: type[TAlgorithm],
    hyperparameters: dict[str, Any],
    preload_state: dict[str, Any],
    batch: ModelInputBatch,
    cell_lines: FeatureDataset,
    drugs: FeatureDataset | None,
) -> TAlgorithm:
    """Configure, train, and return a fitted literature algorithm.

    :param algorithm_cls: Literature algorithm class to instantiate.
    :param hyperparameters: Hyperparameters passed to ``configure``.
    :param preload_state: Attributes to set on the algorithm before training.
    :param batch: Training batch with responses and checkpoint metadata.
    :param cell_lines: Cell-line feature dataset for training.
    :param drugs: Optional drug feature dataset for training.

    :returns: Fitted algorithm instance.

    :raises RuntimeError: If *batch* has no response values.
    """
    if batch.response is None:
        msg = "literature predictor requires response"
        raise RuntimeError(msg)
    algorithm = algorithm_cls()
    for name, value in preload_state.items():
        setattr(algorithm, name, value)
    algorithm.configure(hyperparameters)
    output = DrugResponseDataset(
        response=batch.response,
        cell_line_ids=batch.cell_line_ids,
        drug_ids=batch.drug_ids,
    )
    algorithm.train(
        output,
        cell_lines,
        drugs,
        output_earlystopping=batch.early_stopping_response,
        model_checkpoint_dir=batch.training_context.checkpoint_dir,
    )
    return cast(TAlgorithm, algorithm)


def predict_with_algorithm(
    algorithm: LiteratureTrainingMixin | None,
    batch: ModelInputBatch,
    cell_lines: FeatureDataset,
    drugs: FeatureDataset | None,
) -> Any:
    """Run algorithm prediction or return NaNs when no model is loaded.

    :param algorithm: Fitted algorithm, or ``None`` to emit NaN predictions.
    :param batch: Pairs to score.
    :param cell_lines: Cell-line features aligned with *batch*.
    :param drugs: Optional drug features aligned with *batch*.

    :returns: Predicted responses as a NumPy array.
    """
    import numpy as np

    if algorithm is None:
        return np.full(batch.n_pairs, np.nan, dtype=np.float64)
    return algorithm.predict(
        batch.cell_line_ids,
        batch.drug_ids,
        cell_lines,
        drugs,
    )
