"""Base class for predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureContract
from drevalpy.components.contracts.hyperparameter_space import validate_hyperparameter_space
from drevalpy.log import get_logger
from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.types.enums.prediction_mode import PredictionMode

if TYPE_CHECKING:
    from drevalpy.components.core.batch.model_input_batch import ModelInputBatch

_logger = get_logger(__name__)


class Predictor(ABC):
    """Train and predict drug response from a ``ModelInputBatch``.

    Predictors take featurizer outputs and predict a response for each drug/cell-line pair in the batch.
    Subclasses must be registered to the predictor registry using ``@register_predictor``,
    so that they can be discovered and used in models.
    """

    cell_line_contract: ClassVar[FeatureContract]
    drug_contract: ClassVar[FeatureContract]
    supports_early_stopping: ClassVar[bool] = False
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})
    scope: ClassVar[ModelScope] = ModelScope.MULTI_DRUG
    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ()
    required_drug_blocks: ClassVar[tuple[str, ...]] = ()
    nan_threshold: ClassVar[float] = 0.2

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Reject class-body contract assignments; registration sets them later.

        :param kwargs: Forwarded to ``ABC.__init_subclass__``.
        :raises TypeError: If a contract is assigned on the subclass body.
        """
        super().__init_subclass__(**kwargs)
        forbidden = [name for name in ("cell_line_contract", "drug_contract") if name in cls.__dict__]
        if forbidden:
            msg = (
                f"{cls.__name__}: do not set {', '.join(forbidden)} on the class body; pass them to @register_predictor"
            )
            raise TypeError(msg)

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Store hyperparameters merged with class defaults.

        :param hyperparameters: Optional overrides applied on top of
        """
        self._hyperparameters: dict[str, Any] = {
            **self.get_default_hyperparameters(),
            **(hyperparameters or {}),
        }

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs for HPO.

        :returns: Mapping of parameter name to Ray Tune-style spec dicts.
        """
        return {}

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameter values from the HP space.

        :returns: Parameter names mapped to their declared ``default`` values.
        """
        space = cls.get_hyperparameter_space()
        validate_hyperparameter_space(space, context=f"{cls.__name__}.get_hyperparameter_space()")
        return {key: spec["default"] for key, spec in space.items()}

    def fit(self, batch: ModelInputBatch) -> None:
        """Validate the batch, filter NaN pairs, and delegate to ``_fit``.

        :param batch: Featurized cell-line/drug pairs with training responses.
        :raises ValueError: If *batch* has no response values.
        """
        if batch.response is None:
            msg = "Predictors require response values during fit"
            raise ValueError(msg)
        valid_mask = self._valid_pair_mask(batch)
        self._warn_if_above_threshold(valid_mask, f"{type(self).__name__}.fit")
        if valid_mask.all():
            self._fit(batch)
        else:
            self._fit(batch.subset_pairs(valid_mask))

    @abstractmethod
    def _fit(self, batch: ModelInputBatch) -> None:
        """Subclass fitting logic (response is guaranteed non-None).

        :param batch: Featurized cell-line/drug pairs with training responses.
        """

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict response, returning NaN for pairs with NaN features.

        :param batch: Featurized cell-line/drug pairs to score.

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

        :param batch: Featurized cell-line/drug pairs to score.

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
        cl_feats = batch.cell_line_features
        cl_pair_idx = batch.cell_line_pair_idx
        valid = np.ones(batch.n_pairs, dtype=bool)

        if cl_feats.size > 0 and cl_feats.dtype.kind == "f":
            pair_cl = cl_feats[cl_pair_idx]
            valid &= ~np.isnan(pair_cl).any(axis=1)

        if batch.drug_features is not None and batch.drug_features.size > 0 and batch.drug_features.dtype.kind == "f":
            drug_pair_idx = batch.drug_pair_idx
            if drug_pair_idx is not None:
                pair_dr = batch.drug_features[drug_pair_idx]
                valid &= ~np.isnan(pair_dr).any(axis=1)

        return valid

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

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state for checkpoint persistence.

        :returns: JSON-serializable mapping of fitted attributes.
        """
        return {}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore fitted state from a checkpoint produced by ``get_state``.

        :param state: Mapping previously returned by ``get_state``.
        """
        _ = state

    def is_fitted(self) -> bool:
        """Return whether the predictor has been fit.

        :returns: ``True`` when ``get_state`` returns a non-empty mapping.
        """
        return bool(self.get_state())
