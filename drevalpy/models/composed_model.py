"""Training and prediction for composed featurizer/predictor models."""

from __future__ import annotations

from typing import Any

import numpy as np

from drevalpy.components.featurizers._matrix import unique_entity_ids
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.model_input_build import build_model_input_batch
from drevalpy.components.predictors.base import Predictor
from drevalpy.components.training_context import TrainingContext
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.config import ModelConfig, PredictionMode


def _matrix_feature_width(matrix: np.ndarray | None) -> int:
    """Return the feature width of a featurizer matrix, including object arrays."""
    if matrix is None or matrix.size == 0:
        return 0
    if matrix.dtype == object:
        first = matrix.reshape(-1)[0]
        if hasattr(first, "num_node_features"):
            return int(first.num_node_features)
        first_array = np.asarray(first)
        if first_array.ndim == 0:
            return int(first_array.size)
        return int(first_array.shape[-1])
    if matrix.ndim == 1:
        return int(matrix.shape[0])
    return int(matrix.shape[1])


def _entity_id_only_featurizer(featurizer: Featurizer | None) -> bool:
    return getattr(featurizer, "entity_id_only", False)


def _empty_feature_dataset() -> FeatureDataset:
    return FeatureDataset(features={})


class ComposedModel:
    """Fit featurizers on training entities, then train a predictor on featurized pairs."""

    def __init__(
        self,
        cell_line_featurizer: Featurizer | None,
        drug_featurizer: Featurizer | None,
        predictor: Predictor,
        *,
        predictor_hyperparameters: dict[str, Any] | None = None,
        prediction_mode: PredictionMode = PredictionMode.REGRESSION,
        config: ModelConfig | None = None,
    ) -> None:
        self._cell_line_featurizer = cell_line_featurizer
        self._drug_featurizer = drug_featurizer
        self._predictor = predictor
        self._predictor_hp = predictor_hyperparameters or {}
        self._prediction_mode = prediction_mode
        self._config = config.model_copy(deep=True) if config is not None else None
        self._cell_line_matrix: np.ndarray | None = None
        self._drug_matrix: np.ndarray | None = None
        self._cell_line_entity_ids: np.ndarray | None = None
        self._drug_entity_ids: np.ndarray | None = None

    def _merged_hyperparameters(self) -> dict[str, Any]:
        return {
            **self._predictor.get_default_hyperparameters(),
            **self._predictor_hp,
            "prediction_mode": self._prediction_mode,
        }

    def _input_dims(
        self,
        cell_line_matrix: np.ndarray,
        drug_matrix: np.ndarray | None,
    ) -> dict[str, Any]:
        return {
            "cell_line": _matrix_feature_width(cell_line_matrix),
            "drug": _matrix_feature_width(drug_matrix),
            "n_classes": 1,
        }

    def _build_batch(
        self,
        response: DrugResponseDataset,
        *,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None,
        cell_line_entity_ids: np.ndarray,
        drug_entity_ids: np.ndarray | None,
        cell_line_matrix: np.ndarray,
        drug_matrix: np.ndarray | None,
        output_earlystopping: DrugResponseDataset | None = None,
        training_context: TrainingContext | None = None,
    ) -> ModelInputBatch:
        cell_line_blocks: dict[str, np.ndarray] = {}
        if self._cell_line_featurizer is not None:
            cell_line_blocks = self._cell_line_featurizer.transform_blocks(
                cell_line_input,
                cell_line_entity_ids,
            )

        drug_blocks: dict[str, np.ndarray] = {}
        if self._drug_featurizer is not None and drug_entity_ids is not None:
            drug_source = drug_input if drug_input is not None else _empty_feature_dataset()
            if drug_input is None and not _entity_id_only_featurizer(self._drug_featurizer):
                msg = "drug_input is required when a drug featurizer is configured"
                raise ValueError(msg)
            drug_blocks = self._drug_featurizer.transform_blocks(drug_source, drug_entity_ids)

        return build_model_input_batch(
            response,
            cell_line_entity_ids=cell_line_entity_ids,
            drug_entity_ids=drug_entity_ids if self._drug_featurizer is not None else None,
            cell_line_features=cell_line_matrix,
            drug_features=drug_matrix if self._drug_featurizer is not None else None,
            cell_line_blocks=cell_line_blocks,
            drug_blocks=drug_blocks,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            early_stopping_response=output_earlystopping,
            training_context=training_context,
        )

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        *,
        output_earlystopping: DrugResponseDataset | None = None,
        training_context: TrainingContext | None = None,
    ) -> ComposedModel:
        if len(output) == 0:
            return self

        train_cell_lines = unique_entity_ids(output.cell_line_ids)
        train_drugs = unique_entity_ids(output.drug_ids)

        if self._cell_line_featurizer is not None:
            cl_source = cell_line_input
            self._cell_line_featurizer.fit(cl_source, entity_ids=train_cell_lines)
            if _entity_id_only_featurizer(self._cell_line_featurizer):
                self._cell_line_entity_ids = np.asarray(train_cell_lines, dtype=str)
            else:
                self._cell_line_entity_ids = np.array(list(cell_line_input.features.keys()), dtype=str)
            self._cell_line_matrix = self._cell_line_featurizer.transform(cl_source, self._cell_line_entity_ids)
        else:
            self._cell_line_entity_ids = np.array([], dtype=str)
            self._cell_line_matrix = np.empty((0, 0), dtype=np.float32)

        if self._drug_featurizer is not None:
            drug_source = drug_input if drug_input is not None else _empty_feature_dataset()
            if drug_input is None and not _entity_id_only_featurizer(self._drug_featurizer):
                msg = "drug_input is required when a drug featurizer is configured"
                raise ValueError(msg)
            self._drug_featurizer.fit(drug_source, entity_ids=train_drugs)
            if _entity_id_only_featurizer(self._drug_featurizer):
                self._drug_entity_ids = np.asarray(train_drugs, dtype=str)
            elif drug_input is not None:
                self._drug_entity_ids = np.array(list(drug_input.features.keys()), dtype=str)
            else:
                self._drug_entity_ids = np.asarray(train_drugs, dtype=str)
            self._drug_matrix = self._drug_featurizer.transform(
                drug_source,
                self._drug_entity_ids,
            )
        else:
            self._drug_entity_ids = np.array([], dtype=str)
            self._drug_matrix = np.empty((0, 0), dtype=np.float32)

        batch = self._build_batch(
            output,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            cell_line_entity_ids=self._cell_line_entity_ids,
            drug_entity_ids=self._drug_entity_ids,
            cell_line_matrix=self._cell_line_matrix,
            drug_matrix=self._drug_matrix,
            output_earlystopping=output_earlystopping,
            training_context=training_context,
        )
        self._predictor.build(
            self._merged_hyperparameters(),
            self._input_dims(self._cell_line_matrix, self._drug_matrix),
        )
        self._predictor.fit(batch)
        return self

    @property
    def config(self) -> ModelConfig | None:
        """Return a defensive copy of the resolved model config."""
        return self._config.model_copy(deep=True) if self._config is not None else None

    def is_fitted(self) -> bool:
        """Return whether the predictor has fitted state."""
        return self._predictor.is_fitted()

    def update_predictor_hyperparameters(self, updates: dict[str, Any]) -> None:
        """Merge predictor hyperparameters into the live stack and stored config."""
        filtered = {key: value for key, value in updates.items() if key not in {"cell_line_views", "drug_views"}}
        if not filtered:
            return
        self._predictor_hp.update(filtered)
        if self._config is None:
            return
        self._config = self._config.model_copy(
            update={
                "predictor": self._config.predictor.model_copy(
                    update={
                        "hyperparameters": {
                            **self._config.predictor.hyperparameters,
                            **filtered,
                        }
                    },
                    deep=True,
                )
            },
            deep=True,
        )

    def component_state(self) -> dict[str, object]:
        """Return serializable state owned by the component stack."""
        return {
            "predictor": self._predictor.get_state(),
            "cell_line_featurizer": (
                self._cell_line_featurizer.get_state() if self._cell_line_featurizer is not None else {}
            ),
            "drug_featurizer": self._drug_featurizer.get_state() if self._drug_featurizer is not None else {},
        }

    def restore_component_state(self, state: dict[str, object]) -> None:
        """Restore state produced by ``component_state``."""
        predictor_state = state.get("predictor", {})
        if not isinstance(predictor_state, dict):
            raise ValueError("predictor state is not a mapping")
        self._predictor.set_state(predictor_state)
        for key, featurizer in (
            ("cell_line_featurizer", self._cell_line_featurizer),
            ("drug_featurizer", self._drug_featurizer),
        ):
            value = state.get(key, {})
            if featurizer is not None:
                if not isinstance(value, dict):
                    raise ValueError(f"{key} state is not a mapping")
                featurizer.set_state(value)

    def save(self, directory: str) -> None:
        """Persist config and fitted component state."""
        from drevalpy.models._component_persistence import save_composed_model

        save_composed_model(self, directory)

    @classmethod
    def load(cls, directory: str) -> ComposedModel:
        """Load the canonical native component-stack format."""
        from drevalpy.models._component_persistence import load_composed_model

        return load_composed_model(directory)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        if not self.is_fitted():
            msg = "Model has not been trained; call train() or load() before predict()"
            raise RuntimeError(msg)
        if len(cell_line_ids) == 0:
            return np.array([])

        cell_line_entity_ids = np.array([], dtype=str)
        cell_line_matrix = np.empty((0, 0), dtype=np.float32)
        if self._cell_line_featurizer is not None:
            cell_line_entity_ids = unique_entity_ids(cell_line_ids)
            cell_line_matrix = self._cell_line_featurizer.transform(cell_line_input, cell_line_entity_ids)

        drug_entity_ids: np.ndarray | None = None
        drug_matrix: np.ndarray | None = None
        if self._drug_featurizer is not None:
            drug_entity_ids = unique_entity_ids(drug_ids)
            drug_source = drug_input if drug_input is not None else _empty_feature_dataset()
            if drug_input is None and not _entity_id_only_featurizer(self._drug_featurizer):
                msg = "drug_input is required when a drug featurizer is configured"
                raise ValueError(msg)
            drug_matrix = self._drug_featurizer.transform(drug_source, drug_entity_ids)

        response = DrugResponseDataset(
            response=np.zeros(len(cell_line_ids)),
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
        )
        batch = self._build_batch(
            response,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            cell_line_entity_ids=cell_line_entity_ids,
            drug_entity_ids=drug_entity_ids,
            cell_line_matrix=cell_line_matrix,
            drug_matrix=drug_matrix,
        )
        return self._predictor.predict(batch)
