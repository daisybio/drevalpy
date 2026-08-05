"""pharmaformer literature predictor."""

from __future__ import annotations

from typing import Any, ClassVar, cast

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import BlockSpec
from drevalpy.components.predictors.feature_dataset_block import FeatureDatasetBlockPredictor
from drevalpy.components.predictors.literature._metadata import PHARMAFORMER_REFERENCE
from drevalpy.components.predictors.literature._training_helpers import LiteratureTrainingMixin
from drevalpy.components.predictors.literature.pharmaformer.algorithm import PharmaFormerModel
from drevalpy.components.predictors.literature.pharmaformer.state import apply_state, export_state
from drevalpy.components.registry import register_predictor
from drevalpy.models.config import PredictionMode


@register_predictor(
    "pharmaFormer",
    description="PharmaFormer landmark genes + BPE PharmaFormer model.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=PHARMAFORMER_REFERENCE,
)
class PharmaFormerPredictor(FeatureDatasetBlockPredictor):
    """Registered pharmaformer predictor."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("gene_expression",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("bpe_smiles",)
    required_cell_line_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
    )
    required_drug_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("bpe_smiles", FeatureFormat.NUMERIC_MATRIX),
    )
    requires_drug_featurizer: ClassVar[bool] = True
    validate_drug_graphs: ClassVar[bool] = False
    supports_early_stopping: ClassVar[bool] = True
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    @property
    def _algorithm_cls(self) -> type[LiteratureTrainingMixin]:
        return PharmaFormerModel

    def _export_algorithm_state(self, algorithm: LiteratureTrainingMixin) -> dict[str, Any]:
        return export_state(cast(PharmaFormerModel, algorithm))

    def _apply_algorithm_state(self, payload: dict[str, Any]) -> LiteratureTrainingMixin:
        return apply_state(payload)

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameters from the algorithm class.

        :returns: Default hyperparameter mapping.
        """
        return dict(PharmaFormerModel.get_default_hyperparameters())

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space when exposed by the algorithm.

        :returns: Ray Tune-style hyperparameter specs.
        """
        space = getattr(PharmaFormerModel, "get_hyperparameter_space", None)
        if callable(space):
            return dict(space())
        return {}
