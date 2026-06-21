"""Naive baseline predictors."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.pair_context import PairContext
from drevalpy.components.config import PredictionMode
from drevalpy.components.predictors.baseline import BaselinePredictor
from drevalpy.components.registry import register_predictor


def _mode_value(y: np.ndarray) -> float:
    values, counts = np.unique(y, return_counts=True)
    return float(values[int(np.argmax(counts))])


@register_predictor(
    "naiveMean",
    description="Predict the global mean response.",
    category="baseline",
)
class NaiveMeanPredictor(BaselinePredictor):
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self) -> None:
        self._dataset_mean: float | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        pair_context: PairContext | None = None,
    ) -> None:
        _ = x, pair_context
        self._dataset_mean = float(np.mean(y))

    def predict(
        self,
        x: np.ndarray,
        *,
        pair_context: PairContext | None = None,
    ) -> np.ndarray:
        _ = x, pair_context
        if self._dataset_mean is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        n = len(pair_context.cell_line_ids) if pair_context is not None else len(x)
        return np.full(n, self._dataset_mean, dtype=np.float64)


class _SingleEntityNaivePredictor(BaselinePredictor):
    def __init__(self) -> None:
        self._dataset_mean: float | None = None
        self._entity_means: dict[str, float] = {}

    def _entity_keys(self, pair_context: PairContext) -> np.ndarray:
        raise NotImplementedError

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        pair_context: PairContext | None = None,
    ) -> None:
        _ = x
        if pair_context is None:
            msg = "Naive predictors require pair_context"
            raise ValueError(msg)
        self._dataset_mean = float(np.mean(y))
        keys = self._entity_keys(pair_context)
        for entity in np.unique(keys.astype(str)):
            mask = keys.astype(str) == entity
            self._entity_means[str(entity)] = float(np.mean(y[mask]))

    def predict(
        self,
        x: np.ndarray,
        *,
        pair_context: PairContext | None = None,
    ) -> np.ndarray:
        _ = x
        if pair_context is None or self._dataset_mean is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        keys = self._entity_keys(pair_context).astype(str)
        return np.array(
            [self._entity_means.get(key, self._dataset_mean) for key in keys],
            dtype=np.float64,
        )


@register_predictor(
    "naiveDrugMean",
    description="Predict per-drug mean response with global fallback.",
    category="baseline",
)
class NaiveDrugMeanPredictor(_SingleEntityNaivePredictor):
    def _entity_keys(self, pair_context: PairContext) -> np.ndarray:
        return pair_context.drug_ids


@register_predictor(
    "naiveCellLineMean",
    description="Predict per-cell-line mean response with global fallback.",
    category="baseline",
)
class NaiveCellLineMeanPredictor(_SingleEntityNaivePredictor):
    def _entity_keys(self, pair_context: PairContext) -> np.ndarray:
        return pair_context.cell_line_ids


@register_predictor(
    "naiveTissueMean",
    description="Predict per-tissue mean response with global fallback.",
    category="baseline",
)
class NaiveTissueMeanPredictor(_SingleEntityNaivePredictor):
    def _entity_keys(self, pair_context: PairContext) -> np.ndarray:
        if pair_context.tissue_ids is None:
            msg = "NaiveTissueMeanPredictor requires tissue_ids in pair_context"
            raise ValueError(msg)
        return pair_context.tissue_ids


@register_predictor(
    "naiveTissueDrugMean",
    description="Predict per tissue-drug combination mean response.",
    category="baseline",
)
class NaiveTissueDrugMeanPredictor(BaselinePredictor):
    def __init__(self) -> None:
        self._dataset_mean: float | None = None
        self._combo_means: dict[str, float] = {}

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        pair_context: PairContext | None = None,
    ) -> None:
        _ = x
        if pair_context is None or pair_context.tissue_ids is None:
            msg = "NaiveTissueDrugMeanPredictor requires tissue_ids"
            raise ValueError(msg)
        self._dataset_mean = float(np.mean(y))
        keys = [
            f"{tissue}|{drug}"
            for tissue, drug in zip(
                pair_context.tissue_ids.astype(str),
                pair_context.drug_ids.astype(str),
                strict=True,
            )
        ]
        for combo in np.unique(keys):
            mask = np.array(keys) == combo
            self._combo_means[str(combo)] = float(np.mean(y[mask]))

    def predict(
        self,
        x: np.ndarray,
        *,
        pair_context: PairContext | None = None,
    ) -> np.ndarray:
        _ = x
        if pair_context is None or self._dataset_mean is None or pair_context.tissue_ids is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        keys = [
            f"{tissue}|{drug}"
            for tissue, drug in zip(
                pair_context.tissue_ids.astype(str),
                pair_context.drug_ids.astype(str),
                strict=True,
            )
        ]
        return np.array(
            [self._combo_means.get(key, self._dataset_mean) for key in keys],
            dtype=np.float64,
        )


@register_predictor(
    "naiveMeanEffects",
    description="Predict mean plus cell-line and drug effects.",
    category="baseline",
)
class NaiveMeanEffectsPredictor(BaselinePredictor):
    def __init__(self) -> None:
        self._dataset_mean: float | None = None
        self._cell_line_effects: dict[str, float] = {}
        self._drug_effects: dict[str, float] = {}

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        pair_context: PairContext | None = None,
    ) -> None:
        _ = x
        if pair_context is None:
            msg = "NaiveMeanEffectsPredictor requires pair_context"
            raise ValueError(msg)
        self._dataset_mean = float(np.mean(y))
        for cell_id in np.unique(pair_context.cell_line_ids.astype(str)):
            mask = pair_context.cell_line_ids.astype(str) == cell_id
            self._cell_line_effects[str(cell_id)] = float(np.mean(y[mask]) - self._dataset_mean)
        for drug_id in np.unique(pair_context.drug_ids.astype(str)):
            mask = pair_context.drug_ids.astype(str) == drug_id
            self._drug_effects[str(drug_id)] = float(np.mean(y[mask]) - self._dataset_mean)

    def predict(
        self,
        x: np.ndarray,
        *,
        pair_context: PairContext | None = None,
    ) -> np.ndarray:
        _ = x
        if pair_context is None or self._dataset_mean is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        return np.array(
            [
                self._dataset_mean
                + self._cell_line_effects.get(str(cell_id), 0.0)
                + self._drug_effects.get(str(drug_id), 0.0)
                for cell_id, drug_id in zip(
                    pair_context.cell_line_ids,
                    pair_context.drug_ids,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
