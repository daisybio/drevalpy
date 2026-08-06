"""DrugGNN block literature predictor."""

from __future__ import annotations

from typing import Any, ClassVar, cast

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import BlockSpec
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.feature_dataset_block import FeatureDatasetBlockPredictor
from drevalpy.components.predictors.literature._metadata import DRUGGNN_REFERENCE
from drevalpy.components.predictors.literature._training_helpers import LiteratureTrainingMixin
from drevalpy.components.predictors.literature.druggnn.algorithm import DrugGNN
from drevalpy.components.predictors.literature.druggnn.state import apply_state, export_state
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.registry import register_predictor
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.models.config import PredictionMode


@register_predictor(
    "drugGNN",
    description="DrugGNN: GCN on molecular graphs with dense cell-line features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.GRAPH,
    reference=DRUGGNN_REFERENCE,
)
class DrugGNNPredictor(FeatureDatasetBlockPredictor):
    """Registered DrugGNN predictor."""

    supports_early_stopping: ClassVar[bool] = True
    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("gene_expression",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("drug_graph",)
    required_cell_line_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
    )
    required_drug_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("drug_graph", FeatureFormat.GRAPH),)
    validate_drug_graphs: ClassVar[bool] = True
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    @property
    def _algorithm_cls(self) -> type[LiteratureTrainingMixin]:
        return DrugGNN

    def _export_algorithm_state(self, algorithm: LiteratureTrainingMixin) -> dict[str, Any]:
        return export_state(cast(DrugGNN, algorithm))

    def _apply_algorithm_state(self, payload: dict[str, Any]) -> LiteratureTrainingMixin:
        return apply_state(payload)

    def _materialize_inputs(self, batch: ModelInputBatch) -> tuple[FeatureDataset, FeatureDataset | None]:
        cell_lines, drugs = super()._materialize_inputs(batch)
        if drugs is None:
            raise ValueError("DrugGNN requires drug graph blocks")
        return cell_lines, drugs

    def _is_algorithm_fitted(self, algorithm: LiteratureTrainingMixin | None) -> bool:
        return algorithm is not None and cast(DrugGNN, algorithm).model is not None

    def _validate_restored_payload(self, payload: dict[str, Any]) -> None:
        if payload.get("model_state") is None:
            msg = f"{self.__class__.__name__} payload is missing a trained model"
            raise PredictorStateError(msg)

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameters from the algorithm class.

        :returns: Default hyperparameter mapping.
        """
        defaults = dict(DrugGNN.get_default_hyperparameters())
        defaults.update({"epochs": 2, "batch_size": 8})
        return defaults

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space when exposed by the algorithm.

        :returns: Ray Tune-style hyperparameter specs.
        """
        return {
            "hidden_dim": {"type": "int", "low": 16, "high": 128, "default": 64},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5, "default": 0.2},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True, "default": 1e-3},
            "epochs": {"type": "int", "low": 1, "high": 10, "default": 2},
            "batch_size": {"type": "int", "low": 4, "high": 32, "default": 8},
        }
