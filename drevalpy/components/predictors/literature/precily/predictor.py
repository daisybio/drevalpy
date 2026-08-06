"""Precily block literature predictor."""

from __future__ import annotations

from typing import Any, ClassVar, cast

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.predictors.feature_dataset_block import FeatureDatasetBlockPredictor
from drevalpy.components.predictors.literature._metadata import PRECILY_REFERENCE
from drevalpy.components.predictors.literature._training_helpers import LiteratureTrainingMixin
from drevalpy.components.predictors.literature.precily.algorithm import PrecilyModel
from drevalpy.components.predictors.literature.precily.state import apply_state, export_state
from drevalpy.components.registry import register_predictor
from drevalpy.models.config import PredictionMode


@register_predictor(
    "precily",
    description="Precily pathway + SMILESVec model.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=PRECILY_REFERENCE,
)
class PrecilyPredictor(FeatureDatasetBlockPredictor):
    """Registered Precily predictor."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("pathways",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("smilesvec",)
    supports_early_stopping: ClassVar[bool] = False
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    @property
    def _algorithm_cls(self) -> type[LiteratureTrainingMixin]:
        return PrecilyModel

    def _export_algorithm_state(self, algorithm: LiteratureTrainingMixin) -> dict[str, Any]:
        return export_state(cast(PrecilyModel, algorithm))

    def _apply_algorithm_state(self, payload: dict[str, Any]) -> LiteratureTrainingMixin:
        return apply_state(payload)

    def _is_algorithm_fitted(self, algorithm: LiteratureTrainingMixin | None) -> bool:
        return algorithm is not None and cast(PrecilyModel, algorithm).model is not None

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameters from the algorithm class.

        :returns: Default hyperparameter mapping.
        """
        return dict(PrecilyModel.get_default_hyperparameters())
