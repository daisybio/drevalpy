"""Base class for predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.hyperparameter_space import TunableComponent
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.log import get_logger
from drevalpy.types.enums.accelerator import Accelerator
from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.types.enums.prediction_mode import PredictionMode

if TYPE_CHECKING:
    from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
    from drevalpy.types.data.batch.response_batch import ResponseBatch

_logger = get_logger(__name__)


def fit_response(batch: ModelInputBatch) -> np.ndarray:
    """Return the training responses of *batch*, which :meth:`Predictor.fit` has checked are present.

    :param batch: Batch passed to ``_fit``.
    :returns: ``batch.response``.
    :raises RuntimeError: If the batch has no responses, i.e. ``_fit`` was called directly on a prediction batch.
    """
    if batch.response is None:
        raise RuntimeError("batch.response is required for fit")
    return batch.response


def response_rows(batch: ModelInputBatch, response: ResponseBatch) -> tuple[np.ndarray, np.ndarray | None]:
    """Index the pairs of *response* (usually the early-stopping one) into *batch*'s entity tables.

    :param batch: Batch whose sample and drug feature tables are indexed.
    :param response: Pairs to look up.
    :returns: ``(sample_rows, drug_rows)``; ``drug_rows`` is ``None`` when the batch has no drug features.
    """
    sample_rows = batch.sample.rows_for(response.sample_ids)
    return sample_rows, None if batch.drug is None else batch.drug.rows_for(response.drug_ids)


class Predictor(TunableComponent, ABC):
    """Train and predict drug response from a ``ModelInputBatch``.

    Predictors take featurizer outputs and predict a response for each drug/sample pair in the batch.
    Subclasses must be registered to the predictor registry using ``@register``,
    so that they can be discovered and used in models.

    ``sample_inputs`` and ``drug_inputs`` declare what each side's featurizer must emit
    (see :mod:`drevalpy.components.predictors.inputs`). A fitted predictor is persisted by
    pickling it, so fitted state lives in ordinary attributes.
    """

    sample_inputs: ClassVar[Inputs]
    drug_inputs: ClassVar[Inputs]
    supports_early_stopping: ClassVar[bool] = False
    requires_early_stopping: ClassVar[bool] = False
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})
    scope: ClassVar[ModelScope] = ModelScope.MULTI_DRUG
    nan_threshold: ClassVar[float] = 0.2
    #: Whether it uses a torch accelerator, set by ``register(accelerator=...)``.
    accelerator: ClassVar[Accelerator] = Accelerator.NONE
    #: SPDX license of the ported code, set by ``register(license=...)``; ``None`` means the package's.
    license: ClassVar[str | None] = None
    #: SPDX license of the downloaded third-party files, set by ``register(artifact_license=...)``.
    artifact_license: ClassVar[str | None] = None
    _fitted: bool = False

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Store hyperparameters merged with class defaults.

        :param hyperparameters: Optional overrides applied on top of the defaults from
            ``default_hyperparameters``.
        """
        self._hyperparameters: dict[str, Any] = {
            **self.default_hyperparameters(),
            **(hyperparameters or {}),
        }

    def fit(self, batch: ModelInputBatch) -> None:
        """Validate the batch, filter NaN pairs, and delegate to ``_fit``.

        :param batch: Featurized sample/drug pairs with training responses.
        :raises ValueError: If *batch* has no response values.
        """
        if batch.response is None:
            msg = "Predictors require response values during fit"
            raise ValueError(msg)
        valid_mask = self._valid_pair_mask(batch)
        valid_mask &= ~np.isnan(batch.response)
        self._warn_if_above_threshold(valid_mask, f"{type(self).__name__}.fit")
        if not valid_mask.all():
            batch = batch.subset_pairs(valid_mask)
        self._fit(self._filter_early_stopping(batch))
        self._fitted = True

    def _filter_early_stopping(self, batch: ModelInputBatch) -> ModelInputBatch:
        """Remove unusable stopping pairs and enforce the predictor requirement.

        :param batch: Training batch, possibly carrying an early-stopping response.
        :returns: *batch* with the early-stopping response restricted to pairs with valid features and response.
        :raises ValueError: If the predictor requires early stopping and no usable pair remains.
        """
        response = batch.early_stopping_response
        if response is not None:
            cl_idx, drug_idx = response_rows(batch, response)
            stopping = replace(
                batch,
                sample_ids=response.sample_ids,
                drug_ids=response.drug_ids,
                response=response.response,
                sample_pair_idx=cl_idx,
                drug_pair_idx=drug_idx,
                early_stopping_response=None,
            )
            keep = self._valid_pair_mask(stopping) & ~np.isnan(response.response)
            response = response.subset(keep)
        if self.requires_early_stopping and (response is None or len(response) == 0):
            raise ValueError(f"{type(self).__name__} requires a usable early-stopping holdout")
        return replace(batch, early_stopping_response=response)

    @abstractmethod
    def _fit(self, batch: ModelInputBatch) -> None:
        """Subclass fitting logic (response is guaranteed non-None).

        :param batch: Featurized sample/drug pairs with training responses.
        """

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict response, returning NaN for pairs with NaN features.

        :param batch: Featurized sample/drug pairs to score.

        :returns: One predicted response per pair in *batch*.
        """
        valid_mask = self._valid_pair_mask(batch)
        if valid_mask.all():
            return self._predict(batch)
        result = np.full(batch.n_pairs, np.nan, dtype=np.float64)
        if valid_mask.any():
            result[valid_mask] = self._predict(batch.subset_pairs(valid_mask))
        return result

    @abstractmethod
    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Subclass prediction logic on pre-validated (non-NaN) pairs.

        :param batch: Featurized sample/drug pairs to score.

        :returns: One predicted response per pair in *batch*.
        """

    # ------------------------------------------------------------------
    # NaN detection helpers
    # ------------------------------------------------------------------

    def _valid_pair_mask(self, batch: ModelInputBatch) -> np.ndarray:
        """Return boolean mask over pairs where features are non-NaN.

        :param batch: Input batch.
        :returns: Boolean array of shape ``(batch.n_pairs,)``.
        """
        valid = np.ones(batch.n_pairs, dtype=bool)
        if batch.n_pairs == 0:
            return valid
        for table, indices in ((batch.sample, batch.sample_pair_idx), (batch.drug, batch.drug_pair_idx)):
            if table is None or indices is None:
                continue
            for block in table.blocks.values():
                if block.entity_aligned:
                    valid &= self._valid_feature_rows(block.values[indices])
        return valid

    @staticmethod
    def _valid_feature_rows(rows: np.ndarray) -> np.ndarray:
        """Identify numeric NaNs and object placeholders across trailing dimensions.

        :param rows: Feature rows, one per pair.
        :returns: Boolean array, ``True`` for rows without NaN or ``None`` entries.
        """
        if rows.dtype.kind in "biufc":
            return ~np.isnan(rows).reshape(len(rows), -1).any(axis=1)
        if rows.dtype == object:
            return np.array([all(item is not None for item in row.flat) for row in rows.reshape(len(rows), -1)])
        return np.ones(len(rows), dtype=bool)

    def _warn_if_above_threshold(self, valid_mask: np.ndarray, context: str) -> None:
        """Log a warning when the fraction of invalid pairs exceeds the threshold.

        :param valid_mask: Boolean array (True = valid).
        :param context: Human-readable label for the warning message.
        """
        if len(valid_mask) == 0:
            return
        invalid_frac = 1.0 - valid_mask.mean()
        if invalid_frac > self.nan_threshold:
            _logger.warning(
                "%s: %.0f%% of pairs have NaN features (threshold: %.0f%%)",
                context,
                invalid_frac * 100,
                self.nan_threshold * 100,
            )

    def is_fitted(self) -> bool:
        """Return whether ``fit`` has completed on this predictor.

        :returns: ``True`` after a successful ``fit``.
        """
        return self._fitted
