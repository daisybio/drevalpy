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
    from drevalpy.types.data.predictor_input import PredictorInput
    from drevalpy.types.data.response_table import ResponseTable

_logger = get_logger(__name__)


def get_response(data: PredictorInput) -> np.ndarray:
    """Return the training responses of *data*, which :meth:`Predictor.fit` has checked are present.

    :param data: Input passed to ``_fit``.
    :returns: ``data.response``.
    :raises RuntimeError: If *data* has no responses, i.e. ``_fit`` was called directly on a prediction input.
    """
    if data.response is None:
        raise RuntimeError("data.response is required for fit")
    return data.response


def response_rows(data: PredictorInput, response: ResponseTable) -> tuple[np.ndarray, np.ndarray | None]:
    """Index the pairs of *response* (usually early stopping) into *data*'s entity tables.

    :param data: Input whose sample and drug feature tables are indexed.
    :param response: Pairs to look up.
    :returns: ``(sample_rows, drug_rows)``; ``drug_rows`` is ``None`` when *data* has no drug features.
    """
    sample_rows = data.sample.rows_for(response.sample_ids)
    return sample_rows, None if data.drug is None else data.drug.rows_for(response.drug_ids)


class Predictor(TunableComponent, ABC):
    """Train and predict drug response from a ``PredictorInput``.

    Predictors take featurizer outputs and predict a response for each drug/sample pair in *data*.
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
    #: Response layers of the dataset (such as ``regulation``) the fit reads from
    #: ``data.response_layers``, beside the response it trains on. Training only; a layer the
    #: dataset lacks is left out of the input, so the predictor must cope without it.
    response_layers: ClassVar[tuple[str, ...]] = ()
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

    def fit(self, data: PredictorInput) -> None:
        """Validate the input, filter NaN pairs, and delegate to ``_fit``.

        :param data: Featurized sample/drug pairs with training responses.
        :raises ValueError: If *data* has no response values.
        """
        if data.response is None:
            msg = "Predictors require response values during fit"
            raise ValueError(msg)
        valid_mask = self._valid_pair_mask(data)
        valid_mask &= ~np.isnan(data.response)
        self._warn_if_above_threshold(valid_mask, f"{type(self).__name__}.fit")
        if not valid_mask.all():
            data = data[valid_mask]
        self._fit(self._filter_early_stopping(data))
        self._fitted = True

    def _filter_early_stopping(self, data: PredictorInput) -> PredictorInput:
        """Remove unusable stopping pairs and enforce the predictor requirement.

        :param data: Training input, possibly carrying early-stopping pairs.
        :returns: *data* with early-stopping pairs restricted to pairs with valid features and response.
        :raises ValueError: If the predictor requires early stopping and no usable pair remains.
        """
        response = data.early_stopping
        if response is not None:
            stopping = replace(
                data,
                sample_ids=response.sample_ids,
                drug_ids=response.drug_ids,
                response=response.response,
                early_stopping=None,
                response_layers=response.layers,
            )
            keep = self._valid_pair_mask(stopping) & ~np.isnan(response.response)
            response = response[keep]
        if self.requires_early_stopping and (response is None or len(response) == 0):
            raise ValueError(f"{type(self).__name__} requires a usable early-stopping holdout")
        return replace(data, early_stopping=response)

    @abstractmethod
    def _fit(self, data: PredictorInput) -> None:
        """Subclass fitting logic (response is guaranteed non-None).

        :param data: Featurized sample/drug pairs with training responses.
        """

    def predict(self, data: PredictorInput) -> np.ndarray:
        """Predict response, returning NaN for pairs with NaN features.

        :param data: Featurized sample/drug pairs to score.

        :returns: One predicted response per pair in *data*.
        """
        valid_mask = self._valid_pair_mask(data)
        if valid_mask.all():
            return self._predict(data)
        result = np.full(len(data), np.nan, dtype=np.float64)
        if valid_mask.any():
            result[valid_mask] = self._predict(data[valid_mask])
        return result

    @abstractmethod
    def _predict(self, data: PredictorInput) -> np.ndarray:
        """Subclass prediction logic on pre-validated (non-NaN) pairs.

        :param data: Featurized sample/drug pairs to score.

        :returns: One predicted response per pair in *data*.
        """

    # ------------------------------------------------------------------
    # NaN detection helpers
    # ------------------------------------------------------------------

    def _valid_pair_mask(self, data: PredictorInput) -> np.ndarray:
        """Return boolean mask over pairs where features are non-NaN.

        :param data: Model input.
        :returns: Boolean array of shape ``(len(data),)``.
        """
        valid = np.ones(len(data), dtype=bool)
        if len(data) == 0:
            return valid
        for table, indices in ((data.sample, data.sample_pair_idx), (data.drug, data.drug_pair_idx)):
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
