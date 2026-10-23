"""Base class for predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureContract, normalize_feature_contract
from drevalpy.components.contracts.hyperparameter_space import TunableComponentMixin
from drevalpy.log import get_logger
from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.types.enums.prediction_mode import PredictionMode

if TYPE_CHECKING:
    from drevalpy.types.data.batch.model_input_batch import ModelInputBatch

_logger = get_logger(__name__)


class Predictor(TunableComponentMixin, ABC):
    """Train and predict drug response from a ``ModelInputBatch``.

    Predictors take featurizer outputs and predict a response for each drug/sample pair in the batch.
    Subclasses must be registered to the predictor registry using ``@register``,
    so that they can be discovered and used in models.

    ``sample_contract`` and ``drug_contract`` may be declared on the class body or
    passed to ``@register``. When both are given the decorator argument wins.

    The HPO-space and checkpoint hooks come from ``TunableComponentMixin`` in
    ``contracts/hyperparameter_space.py``, which ``Featurizer`` mixes in as well;
    ``is_fitted`` below is predictor-only and stays here.
    """

    sample_contract: ClassVar[FeatureContract]
    drug_contract: ClassVar[FeatureContract]
    supports_early_stopping: ClassVar[bool] = False
    requires_early_stopping: ClassVar[bool] = False
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})
    scope: ClassVar[ModelScope] = ModelScope.MULTI_DRUG
    required_sample_blocks: ClassVar[tuple[str, ...]] = ()
    required_drug_blocks: ClassVar[tuple[str, ...]] = ()
    nan_threshold: ClassVar[float] = 0.2

    def __init_subclass__(cls, **kwargs: object) -> None:
        r"""Normalize class-body contract declarations, if there are any.

        A ``FeatureFormat`` shorthand is widened to a ``FeatureContract`` so the
        class body and the ``@register`` arguments accept the same spellings.
        Subclasses that declare nothing are registered with the decorator's
        ``sample_contract=`` / ``drug_contract=`` instead.

        :param \*\*kwargs: Forwarded to ``ABC.__init_subclass__``.
        :raises TypeError: If a class-body contract is neither a ``FeatureContract``
            nor a ``FeatureFormat``.
        """
        super().__init_subclass__(**kwargs)
        for attr_name in ("sample_contract", "drug_contract"):
            if attr_name not in cls.__dict__:
                continue
            try:
                setattr(cls, attr_name, normalize_feature_contract(cls.__dict__[attr_name]))
            except TypeError as exc:
                msg = f"{cls.__name__}: class-body {attr_name} is invalid: {exc}"
                raise TypeError(msg) from exc

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Store hyperparameters merged with class defaults.

        :param hyperparameters: Optional overrides applied on top of the defaults from
            ``get_default_hyperparameters``.
        """
        self._hyperparameters: dict[str, Any] = {
            **self.get_default_hyperparameters(),
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

    def _filter_early_stopping(self, batch: ModelInputBatch) -> ModelInputBatch:
        """Remove unusable stopping pairs and enforce the predictor requirement.

        :param batch: Training batch, possibly carrying an early-stopping response.
        :returns: *batch* with the early-stopping response restricted to pairs with valid features and response.
        :raises ValueError: If the predictor requires early stopping and no usable pair remains.
        """
        response = batch.early_stopping_response
        if response is not None:
            cl_idx, drug_idx = batch._pair_indices_for(response)
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
            response = type(response)(response.response[keep], response.sample_ids[keep], response.drug_ids[keep])
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
        """Return whether the predictor has been fit.

        :returns: ``True`` when ``get_state`` returns a non-empty mapping.
        """
        return bool(self.get_state())
