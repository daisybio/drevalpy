"""dipk literature predictor."""

from __future__ import annotations

from typing import Any, ClassVar, cast

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import BlockSpec
from drevalpy.components.predictors.feature_dataset_block import FeatureDatasetBlockPredictor
from drevalpy.components.predictors.literature._metadata import DIPK_REFERENCE
from drevalpy.components.predictors.literature._training_helpers import LiteratureTrainingMixin
from drevalpy.components.predictors.literature.dipk.algorithm import DIPKModel
from drevalpy.components.predictors.literature.dipk.state import apply_state, export_state
from drevalpy.components.registry import register_predictor
from drevalpy.models.config import PredictionMode


@register_predictor(
    "dipk",
    description="DIPK BIONIC + MolGNet model.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.RAGGED_SEQUENCE,
    reference=DIPK_REFERENCE,
)
class DIPKPredictor(FeatureDatasetBlockPredictor):
    """Registered dipk predictor."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("gene_expression", "bionic_features")
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("molgnet_features",)
    required_cell_line_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("bionic_features", FeatureFormat.NUMERIC_MATRIX),
    )
    required_drug_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("molgnet_features", FeatureFormat.RAGGED_SEQUENCE),
    )
    requires_drug_featurizer: ClassVar[bool] = True
    validate_drug_graphs: ClassVar[bool] = False
    supports_early_stopping: ClassVar[bool] = True
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    @property
    def _algorithm_cls(self) -> type[LiteratureTrainingMixin]:
        return DIPKModel

    def _export_algorithm_state(self, algorithm: LiteratureTrainingMixin) -> dict[str, Any]:
        return export_state(cast(DIPKModel, algorithm))

    def _apply_algorithm_state(self, payload: dict[str, Any]) -> LiteratureTrainingMixin:
        return apply_state(payload)

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameters from the algorithm class.

        :returns: Default hyperparameter mapping.
        """
        return dict(DIPKModel.get_default_hyperparameters())

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space when exposed by the algorithm.

        :returns: Ray Tune-style hyperparameter specs.
        """
        space = getattr(DIPKModel, "get_hyperparameter_space", None)
        if callable(space):
            return dict(space())
        return {}
