"""Training and prediction for composed featurizer/predictor models."""

from __future__ import annotations

from typing import Any

import numpy as np

from drevalpy.components.config import PredictionMode
from drevalpy.components.featurizers._matrix import unique_entity_ids
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.pair_context import PairContext
from drevalpy.components.pair_batch_build import build_pair_batch
from drevalpy.components.pair_features import build_pair_matrix
from drevalpy.components.predictors.base import Predictor
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset


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
    ) -> None:
        self._cell_line_featurizer = cell_line_featurizer
        self._drug_featurizer = drug_featurizer
        self._predictor = predictor
        self._predictor_hp = predictor_hyperparameters or {}
        self._prediction_mode = prediction_mode
        self._cell_line_matrix: np.ndarray | None = None
        self._drug_matrix: np.ndarray | None = None
        self._cell_line_entity_ids: np.ndarray | None = None
        self._drug_entity_ids: np.ndarray | None = None
        self._tissue_by_cell_line: dict[str, str] | None = None

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        *,
        tissue_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
    ) -> ComposedModel:
        if len(output) == 0:
            return self

        train_cell_lines = unique_entity_ids(output.cell_line_ids)
        train_drugs = unique_entity_ids(output.drug_ids)

        if self._cell_line_featurizer is not None:
            self._cell_line_featurizer.fit(cell_line_input, entity_ids=train_cell_lines)
            self._cell_line_entity_ids = np.array(list(cell_line_input.features.keys()), dtype=str)
            self._cell_line_matrix = self._cell_line_featurizer.transform(cell_line_input, self._cell_line_entity_ids)
        else:
            self._cell_line_entity_ids = np.array([], dtype=str)
            self._cell_line_matrix = np.empty((0, 0), dtype=np.float32)

        if self._drug_featurizer is not None:
            if drug_input is None:
                msg = "drug_input is required when a drug featurizer is configured"
                raise ValueError(msg)
            self._drug_featurizer.fit(drug_input, entity_ids=train_drugs)
            self._drug_entity_ids = np.array(list(drug_input.features.keys()), dtype=str)
            self._drug_matrix = self._drug_featurizer.transform(
                drug_input,
                self._drug_entity_ids,
            )
        else:
            self._drug_entity_ids = np.array([], dtype=str)
            self._drug_matrix = np.empty((0, 0), dtype=np.float32)

        if tissue_input is not None:
            self._tissue_by_cell_line = {
                str(cell_id): str(tissue_input.features[str(cell_id)]["tissue"][0])
                for cell_id in tissue_input.features
                if "tissue" in tissue_input.features[str(cell_id)]
            }

        if getattr(self._predictor, "uses_structured_features", False):
            cell_line_blocks = (
                self._cell_line_featurizer.transform_blocks(cell_line_input, self._cell_line_entity_ids)
                if self._cell_line_featurizer is not None
                else {}
            )
            drug_blocks: dict[str, np.ndarray] = {}
            if self._drug_featurizer is not None and drug_input is not None:
                drug_blocks = self._drug_featurizer.transform_blocks(
                    drug_input,
                    self._drug_entity_ids,
                )
            batch = build_pair_batch(
                output,
                cell_line_entity_ids=self._cell_line_entity_ids,
                drug_entity_ids=self._drug_entity_ids if self._drug_featurizer is not None else None,
                cell_line_features=self._cell_line_matrix,
                drug_features=self._drug_matrix if self._drug_featurizer is not None else None,
                cell_line_blocks=cell_line_blocks,
                drug_blocks=drug_blocks,
                pair_context=self._pair_context(output.cell_line_ids, output.drug_ids),
            )
            merged_hp = {
                **self._predictor.get_default_hyperparameters(),
                **self._predictor_hp,
                "prediction_mode": self._prediction_mode,
            }
            input_dims = {
                "cell_line": _matrix_feature_width(self._cell_line_matrix),
                "drug": _matrix_feature_width(self._drug_matrix),
                "n_classes": 1,
            }
            self._predictor.build(merged_hp, input_dims)
            self._predictor.fit_structured(
                batch,
                output=output,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
                output_earlystopping=output_earlystopping,
            )
        elif self._predictor.uses_features:
            x = build_pair_matrix(
                output,
                self._cell_line_matrix,
                self._drug_matrix,
                self._cell_line_entity_ids,
                self._drug_entity_ids,
            )
            input_dims = {
                "cell_line": int(self._cell_line_matrix.shape[1]) if self._cell_line_matrix.size else 0,
                "drug": int(self._drug_matrix.shape[1])
                if self._drug_matrix.size
                else 0,
                "n_classes": 1,
            }
            merged_hp = {
                **self._predictor.get_default_hyperparameters(),
                **self._predictor_hp,
                "prediction_mode": self._prediction_mode,
            }
            self._predictor.build(merged_hp, input_dims)
            pair_ctx = self._pair_context(output.cell_line_ids, output.drug_ids)
            self._predictor.fit(x, output.response, pair_context=pair_ctx)
        elif getattr(self._predictor, "uses_raw_features", False):
            merged_hp = {
                **self._predictor.get_default_hyperparameters(),
                **self._predictor_hp,
                "prediction_mode": self._prediction_mode,
            }
            self._predictor.build(merged_hp, {})
            self._predictor.fit_raw(
                output,
                cell_line_input,
                drug_input,
                output_earlystopping=None,
            )
        else:
            merged_hp = {
                **self._predictor.get_default_hyperparameters(),
                **self._predictor_hp,
                "prediction_mode": self._prediction_mode,
            }
            self._predictor.build(merged_hp, {})
            pair_ctx = self._pair_context(output.cell_line_ids, output.drug_ids)
            self._predictor.fit(
                np.empty((len(output), 0)),
                output.response,
                pair_context=pair_ctx,
            )
        return self

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        if len(cell_line_ids) == 0:
            return np.array([])

        if getattr(self._predictor, "uses_structured_features", False):
            cell_line_entity_ids = np.array([], dtype=str)
            cell_line_matrix = np.empty((0, 0), dtype=np.float32)
            cell_line_blocks: dict[str, np.ndarray] = {}
            if self._cell_line_featurizer is not None:
                cell_line_entity_ids = unique_entity_ids(cell_line_ids)
                cell_line_matrix = self._cell_line_featurizer.transform(cell_line_input, cell_line_entity_ids)
                cell_line_blocks = self._cell_line_featurizer.transform_blocks(
                    cell_line_input,
                    cell_line_entity_ids,
                )

            drug_entity_ids = None
            drug_matrix = None
            drug_blocks: dict[str, np.ndarray] = {}
            if self._drug_featurizer is not None:
                if drug_input is None:
                    msg = "drug_input is required when a drug featurizer is configured"
                    raise ValueError(msg)
                drug_entity_ids = unique_entity_ids(drug_ids)
                drug_matrix = self._drug_featurizer.transform(drug_input, drug_entity_ids)
                drug_blocks = self._drug_featurizer.transform_blocks(drug_input, drug_entity_ids)

            response = DrugResponseDataset(
                response=np.zeros(len(cell_line_ids)),
                cell_line_ids=cell_line_ids,
                drug_ids=drug_ids,
            )
            batch = build_pair_batch(
                response,
                cell_line_entity_ids=cell_line_entity_ids,
                drug_entity_ids=drug_entity_ids,
                cell_line_features=cell_line_matrix,
                drug_features=drug_matrix,
                cell_line_blocks=cell_line_blocks,
                drug_blocks=drug_blocks,
                pair_context=self._pair_context(cell_line_ids, drug_ids, cell_line_input=cell_line_input),
            )
            return self._predictor.predict_structured(
                batch,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
            )

        if not self._predictor.uses_features:
            if getattr(self._predictor, "uses_raw_features", False):
                return self._predictor.predict_raw(
                    cell_line_ids,
                    drug_ids,
                    cell_line_input,
                    drug_input,
                )
            pair_ctx = self._pair_context(cell_line_ids, drug_ids, cell_line_input=cell_line_input)
            return self._predictor.predict(
                np.empty((len(cell_line_ids), 0)),
                pair_context=pair_ctx,
            )

        if self._cell_line_featurizer is None and self._drug_featurizer is None:
            msg = "Call train() before predict()"
            raise RuntimeError(msg)

        cell_line_entity_ids = np.array([], dtype=str)
        cell_line_matrix = np.empty((0, 0), dtype=np.float32)
        if self._cell_line_featurizer is not None:
            cell_line_entity_ids = unique_entity_ids(cell_line_ids)
            cell_line_matrix = self._cell_line_featurizer.transform(cell_line_input, cell_line_entity_ids)

        drug_entity_ids = np.array([], dtype=str)
        drug_matrix = np.empty((0, 0), dtype=np.float32)
        if self._drug_featurizer is not None:
            if drug_input is None:
                msg = "drug_input is required when a drug featurizer is configured"
                raise ValueError(msg)
            drug_entity_ids = unique_entity_ids(drug_ids)
            drug_matrix = self._drug_featurizer.transform(
                drug_input,
                drug_entity_ids,
            )

        response = DrugResponseDataset(
            response=np.zeros(len(cell_line_ids)),
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
        )
        x = build_pair_matrix(
            response,
            cell_line_matrix,
            drug_matrix,
            cell_line_entity_ids,
            drug_entity_ids,
        )
        return self._predictor.predict(x, pair_context=self._pair_context(cell_line_ids, drug_ids, cell_line_input=cell_line_input))

    def _pair_context(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        *,
        cell_line_input: FeatureDataset | None = None,
    ) -> PairContext:
        tissue_ids = None
        if self._tissue_by_cell_line is not None:
            tissue_ids = np.array(
                [self._tissue_by_cell_line.get(str(cl), "") for cl in cell_line_ids],
                dtype=object,
            )
        elif cell_line_input is not None:
            from drevalpy.datasets.utils import TISSUE_IDENTIFIER

            if any(TISSUE_IDENTIFIER in views for views in cell_line_input.features.values()):
                tissue_ids = np.asarray(
                    cell_line_input.get_feature_matrix(
                        view=TISSUE_IDENTIFIER,
                        identifiers=cell_line_ids,
                    )
                ).reshape(-1)
        return PairContext(cell_line_ids=cell_line_ids, drug_ids=drug_ids, tissue_ids=tissue_ids)
