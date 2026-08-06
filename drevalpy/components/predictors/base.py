"""Base class for predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureContract
from drevalpy.types.model_scope import ModelScope
from drevalpy.types.prediction_mode import PredictionMode

if TYPE_CHECKING:
    from drevalpy.components.model_input_batch import ModelInputBatch


class Predictor(ABC):
    """Train and predict drug response from a ``ModelInputBatch``.

    Predictors take featurizer outputs and predict a response for each drug/cell-line pair in the batch.
    Subclasses must be registered to the predictor registry using ``@register_predictor``,
    so that they can be discovered and used in models.
    """

    cell_line_contract: ClassVar[FeatureContract]
    drug_contract: ClassVar[FeatureContract]
    requires_drug_featurizer: ClassVar[bool] = True
    routing_drug_featurizer: ClassVar[str | None] = None
    supports_early_stopping: ClassVar[bool] = False
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})
    supported_scopes: ClassVar[frozenset[ModelScope]] = frozenset({ModelScope.MULTI_DRUG})
    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ()
    required_drug_blocks: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Reject class-body contract assignments; registration sets them later.

        :param kwargs: Forwarded to ``ABC.__init_subclass__``.
        :raises TypeError: If a contract is assigned on the subclass body.
        """
        super().__init_subclass__(**kwargs)
        forbidden = [name for name in ("cell_line_contract", "drug_contract") if name in cls.__dict__]
        if forbidden:
            msg = (
                f"{cls.__name__}: do not set {', '.join(forbidden)} on the class body; "
                "pass them to @register_predictor"
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
        from drevalpy.components.hyperparameter_space import validate_hyperparameter_space

        space = cls.get_hyperparameter_space()
        validate_hyperparameter_space(space, context=f"{cls.__name__}.get_hyperparameter_space()")
        return {key: spec["default"] for key, spec in space.items()}

    @abstractmethod
    def fit(self, batch: ModelInputBatch) -> None:
        """Fit on a featurized predictor input batch.

        :param batch: Featurized cell-line/drug pairs with training responses.
        """

    @abstractmethod
    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict response for each pair in the batch.

        :param batch: Featurized cell-line/drug pairs to score.

        :returns: One predicted response per pair in *batch*.
        """

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state for legacy save/load bridges.

        :returns: JSON-serializable mapping of fitted attributes.
        """
        return {}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore fitted state produced by ``get_state``.

        :param state: Mapping previously returned by ``get_state``.
        """
        _ = state

    def is_fitted(self) -> bool:
        """Return whether the predictor has been fit.

        :returns: ``True`` when ``get_state`` returns a non-empty mapping.
        """
        return bool(self.get_state())
