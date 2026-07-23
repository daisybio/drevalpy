"""Naive mean-effects predictor."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors._identity_batch import pair_tissue_ids
from drevalpy.components.predictors.structured import BlockPredictor
from drevalpy.components.registry import register_predictor
from drevalpy.components.state_helpers import state_float, state_str_dict


@register_predictor(
    "naiveMeanEffects",
    description="Predict mean plus cell-line and drug effects.",
    category="baseline",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
)
class NaiveMeanEffectsPredictor(BlockPredictor):
    """Naive mean effects predictor component."""

    requires_drug_featurizer: ClassVar[bool] = False

    def __init__(self) -> None:
        self._dataset_mean: float | None = None
        self._tissue_effects: dict[str, float] = {}
        self._cell_line_effects: dict[str, float] = {}
        self._drug_effects: dict[str, float] = {}

    def fit(self, batch: ModelInputBatch) -> None:
        if batch.response is None:
            msg = "Naive predictors require response values during fit"
            raise ValueError(msg)
        y = np.asarray(batch.response, dtype=np.float64)
        self._dataset_mean = float(np.mean(y))
        cell_line_ids = batch.cell_line_ids.astype(str)
        drug_ids = batch.drug_ids.astype(str)

        cell_line_means: dict[str, float] = {}
        for cell_id in np.unique(cell_line_ids):
            mask = cell_line_ids == cell_id
            cell_line_means[str(cell_id)] = float(np.mean(y[mask]))

        tissue_ids = pair_tissue_ids(batch, cell_line_input=batch.cell_line_input)
        if tissue_ids is not None:
            tissue_means: dict[str, float] = {}
            for tissue in np.unique(tissue_ids.astype(str)):
                mask = tissue_ids.astype(str) == tissue
                tissue_means[str(tissue)] = float(np.mean(y[mask]))

            self._tissue_effects = {tissue: mean - self._dataset_mean for tissue, mean in tissue_means.items()}

            cell_line_to_tissue: dict[str, str] = {}
            for cell_id in np.unique(cell_line_ids):
                mask = cell_line_ids == cell_id
                tissue = tissue_ids[mask][0]
                cell_line_to_tissue[str(cell_id)] = str(tissue)

            self._cell_line_effects = {
                cell_id: cell_line_means[cell_id] - tissue_means[cell_line_to_tissue[cell_id]]
                for cell_id in cell_line_means
            }
        else:
            self._tissue_effects = {}
            self._cell_line_effects = {cell_id: mean - self._dataset_mean for cell_id, mean in cell_line_means.items()}

        for drug_id in np.unique(drug_ids):
            mask = drug_ids == drug_id
            self._drug_effects[str(drug_id)] = float(np.mean(y[mask]) - self._dataset_mean)

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        if self._dataset_mean is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        tissue_ids = pair_tissue_ids(batch, cell_line_input=batch.cell_line_input)
        if self._tissue_effects and tissue_ids is not None:
            return np.array(
                [
                    self._dataset_mean
                    + self._tissue_effects.get(str(tissue), 0.0)
                    + self._cell_line_effects.get(str(cell_id), 0.0)
                    + self._drug_effects.get(str(drug_id), 0.0)
                    for cell_id, drug_id, tissue in zip(
                        batch.cell_line_ids,
                        batch.drug_ids,
                        tissue_ids,
                        strict=True,
                    )
                ],
                dtype=np.float64,
            )
        return np.array(
            [
                self._dataset_mean
                + self._cell_line_effects.get(str(cell_id), 0.0)
                + self._drug_effects.get(str(drug_id), 0.0)
                for cell_id, drug_id in zip(batch.cell_line_ids, batch.drug_ids, strict=True)
            ],
            dtype=np.float64,
        )

    def get_state(self) -> dict[str, object]:
        if self._dataset_mean is None:
            return {}
        return {
            "dataset_mean": self._dataset_mean,
            "tissue_effects": dict(self._tissue_effects),
            "cell_line_effects": dict(self._cell_line_effects),
            "drug_effects": dict(self._drug_effects),
        }

    def set_state(self, state: dict[str, object]) -> None:
        mean = state_float(state, "dataset_mean")
        if mean is not None:
            self._dataset_mean = mean
        tissue_effects = state_str_dict(state, "tissue_effects")
        if tissue_effects:
            self._tissue_effects = tissue_effects
        cell_line_effects = state_str_dict(state, "cell_line_effects")
        if cell_line_effects:
            self._cell_line_effects = cell_line_effects
        drug_effects = state_str_dict(state, "drug_effects")
        if drug_effects:
            self._drug_effects = drug_effects

    def is_fitted(self) -> bool:
        return self._dataset_mean is not None
