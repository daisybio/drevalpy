"""SRMF block literature predictor."""

from __future__ import annotations

from typing import Any, ClassVar, cast

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.predictors.feature_dataset_block import FeatureDatasetBlockPredictor
from drevalpy.components.predictors.literature._metadata import SRMF_REFERENCE
from drevalpy.components.predictors.literature._training_helpers import LiteratureTrainingMixin
from drevalpy.components.predictors.literature.srmf.algorithm import SRMF
from drevalpy.components.predictors.literature.srmf.state import apply_state, export_state
from drevalpy.components.registry import register_predictor
from drevalpy.models.config import PredictionMode


@register_predictor(
    "srmf",
    description="SRMF matrix factorization model.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=SRMF_REFERENCE,
)
class SRMFPredictor(FeatureDatasetBlockPredictor):
    """Registered SRMF predictor."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("gene_expression",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("fingerprints",)
    requires_drug_featurizer: ClassVar[bool] = True
    supports_early_stopping: ClassVar[bool] = False
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    @property
    def _algorithm_cls(self) -> type[LiteratureTrainingMixin]:
        return SRMF

    def _export_algorithm_state(self, algorithm: LiteratureTrainingMixin) -> dict[str, Any]:
        return export_state(cast(SRMF, algorithm))

    def _apply_algorithm_state(self, payload: dict[str, Any]) -> LiteratureTrainingMixin:
        return apply_state(payload)

    def _is_algorithm_fitted(self, algorithm: LiteratureTrainingMixin | None) -> bool:
        return algorithm is not None and not cast(SRMF, algorithm).best_u.empty

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameters from the algorithm class.

        :returns: Default hyperparameter mapping.
        """
        return dict(SRMF.get_default_hyperparameters())

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space when exposed by the algorithm.

        :returns: Ray Tune-style hyperparameter specs.
        """
        space = getattr(SRMF, "get_hyperparameter_space", None)
        if callable(space):
            return dict(space())
        return {}
