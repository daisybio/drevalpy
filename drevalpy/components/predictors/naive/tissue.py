"""Tissue-aware naive mean predictors."""

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
    "naiveTissueMean",
    description="Predict per-tissue mean response with global fallback.",
    category="baseline",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
)
class NaiveTissueMeanPredictor(BlockPredictor):
    """Naive tissue mean predictor component."""

    requires_drug_featurizer: ClassVar[bool] = False

    def __init__(self) -> None:
        self._dataset_mean: float | None = None
        self._entity_means: dict[str, float] = {}

    def fit(self, batch: ModelInputBatch) -> None:
        if batch.response is None:
            msg = "Naive predictors require response values during fit"
            raise ValueError(msg)
        tissue_ids = pair_tissue_ids(batch, cell_line_input=batch.cell_line_input)
        if tissue_ids is None:
            msg = "NaiveTissueMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        y = np.asarray(batch.response, dtype=np.float64)
        self._dataset_mean = float(np.mean(y))
        for tissue in np.unique(tissue_ids.astype(str)):
            mask = tissue_ids.astype(str) == tissue
            self._entity_means[str(tissue)] = float(np.mean(y[mask]))

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        if self._dataset_mean is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        tissue_ids = pair_tissue_ids(batch, cell_line_input=batch.cell_line_input)
        if tissue_ids is None:
            msg = "NaiveTissueMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        return np.array(
            [self._entity_means.get(str(tissue), self._dataset_mean) for tissue in tissue_ids],
            dtype=np.float64,
        )

    def get_state(self) -> dict[str, object]:
        if self._dataset_mean is None:
            return {}
        return {
            "dataset_mean": self._dataset_mean,
            "tissue_means": dict(self._entity_means),
        }

    def set_state(self, state: dict[str, object]) -> None:
        mean = state_float(state, "dataset_mean")
        if mean is not None:
            self._dataset_mean = mean
        entity_means = state_str_dict(state, "tissue_means")
        if entity_means:
            self._entity_means = entity_means

    def is_fitted(self) -> bool:
        return self._dataset_mean is not None


@register_predictor(
    "naiveTissueDrugMean",
    description="Predict per tissue-drug combination mean response.",
    category="baseline",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
)
class NaiveTissueDrugMeanPredictor(BlockPredictor):
    """Naive tissue drug mean predictor component."""

    requires_drug_featurizer: ClassVar[bool] = False

    def __init__(self) -> None:
        self._dataset_mean: float | None = None
        self._combo_means: dict[str, float] = {}

    def fit(self, batch: ModelInputBatch) -> None:
        if batch.response is None:
            msg = "Naive predictors require response values during fit"
            raise ValueError(msg)
        tissue_ids = pair_tissue_ids(batch, cell_line_input=batch.cell_line_input)
        if tissue_ids is None:
            msg = "NaiveTissueDrugMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        y = np.asarray(batch.response, dtype=np.float64)
        self._dataset_mean = float(np.mean(y))
        keys = [
            f"{tissue}|{drug}" for tissue, drug in zip(tissue_ids.astype(str), batch.drug_ids.astype(str), strict=True)
        ]
        for combo in np.unique(keys):
            mask = np.array(keys) == combo
            self._combo_means[str(combo)] = float(np.mean(y[mask]))

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        if self._dataset_mean is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        tissue_ids = pair_tissue_ids(batch, cell_line_input=batch.cell_line_input)
        if tissue_ids is None:
            msg = "NaiveTissueDrugMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        keys = [
            f"{tissue}|{drug}" for tissue, drug in zip(tissue_ids.astype(str), batch.drug_ids.astype(str), strict=True)
        ]
        return np.array(
            [self._combo_means.get(key, self._dataset_mean) for key in keys],
            dtype=np.float64,
        )

    def get_state(self) -> dict[str, object]:
        if self._dataset_mean is None:
            return {}
        return {
            "dataset_mean": self._dataset_mean,
            "tissue_drug_means": {tuple(key.split("|", maxsplit=1)): mean for key, mean in self._combo_means.items()},
        }

    def set_state(self, state: dict[str, object]) -> None:
        mean = state_float(state, "dataset_mean")
        if mean is not None:
            self._dataset_mean = mean
        combo_means = state.get("tissue_drug_means")
        if isinstance(combo_means, dict):
            self._combo_means = {
                f"{tissue}|{drug}": float(value)
                for (tissue, drug), value in combo_means.items()
                if isinstance(tissue, str) and isinstance(drug, str)
            }

    def is_fitted(self) -> bool:
        return self._dataset_mean is not None
