"""sparsego literature predictor."""

from __future__ import annotations

from typing import Any, ClassVar, cast

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import BlockSpec
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.feature_dataset_block import FeatureDatasetBlockPredictor
from drevalpy.components.predictors.literature._block_inputs import materialize_block_inputs
from drevalpy.components.predictors.literature._metadata import SPARSEGO_REFERENCE
from drevalpy.components.predictors.literature._training_helpers import LiteratureTrainingMixin
from drevalpy.components.predictors.literature.sparsego.algorithm import SparseGOModel
from drevalpy.components.predictors.literature.sparsego.state import apply_state, export_state
from drevalpy.components.registry import register_predictor
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.models.config import PredictionMode


@register_predictor(
    "sparsego",
    description="SparseGO GO-structured visible neural network.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=SPARSEGO_REFERENCE,
)
class SparseGOPredictor(FeatureDatasetBlockPredictor):
    """Registered sparsego predictor."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ()
    required_cell_line_block_alternatives: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX, metadata=True),
        BlockSpec("mutations", FeatureFormat.NUMERIC_MATRIX, metadata=True),
    )
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("fingerprints",)
    required_drug_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),
    )
    validate_drug_graphs: ClassVar[bool] = False
    supports_early_stopping: ClassVar[bool] = False
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    @property
    def _algorithm_cls(self) -> type[LiteratureTrainingMixin]:
        return SparseGOModel

    def _export_algorithm_state(self, algorithm: LiteratureTrainingMixin) -> dict[str, Any]:
        return export_state(cast(SparseGOModel, algorithm))

    def _apply_algorithm_state(self, payload: dict[str, Any]) -> LiteratureTrainingMixin:
        return apply_state(payload)

    def _materialize_inputs(self, batch: ModelInputBatch) -> tuple[FeatureDataset, FeatureDataset | None]:
        active = [
            spec.name for spec in self.required_cell_line_block_alternatives if spec.name in batch.cell_line_blocks
        ]
        if len(active) != 1:
            raise ValueError(
                "SparseGOPredictor requires exactly one cell-line block from " "['gene_expression', 'mutations']"
            )
        block = batch.cell_line_blocks[active[0]]
        if block.metadata is None:
            raise ValueError("SparseGOPredictor requires ontology metadata on its active cell-line block")
        self._engine_preload_state = dict(block.metadata)
        return materialize_block_inputs(
            self,
            batch,
            required_cell_line_blocks=(active[0],),
            required_drug_blocks=self.required_drug_blocks,
        )

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameters from the algorithm class.

        :returns: Default hyperparameter mapping.
        """
        return dict(SparseGOModel.get_default_hyperparameters())

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space when exposed by the algorithm.

        :returns: Ray Tune-style hyperparameter specs.
        """
        space = getattr(SparseGOModel, "get_hyperparameter_space", None)
        if callable(space):
            return dict(space())
        return {}
