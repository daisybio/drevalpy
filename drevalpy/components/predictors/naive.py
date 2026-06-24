"""Naive baseline predictors."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.pair_context import PairContext
from drevalpy.components.predictors.baseline import BaselinePredictor
from drevalpy.components.registry import register_predictor
from drevalpy.components.state_helpers import state_float, state_str_dict
from drevalpy.models.config import PredictionMode


def _mode_value(y: np.ndarray) -> float:
    values, counts = np.unique(y, return_counts=True)
    return float(values[int(np.argmax(counts))])


@register_predictor(
    "naiveMean",
    description="Predict the global mean response.",
    category="baseline",
)
class NaiveMeanPredictor(BaselinePredictor):
    """Naive mean predictor component."""

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

    def get_state(self) -> dict[str, object]:
        if self._dataset_mean is None:
            return {}
        return {"dataset_mean": self._dataset_mean}

    def set_state(self, state: dict[str, object]) -> None:
        mean = state_float(state, "dataset_mean")
        if mean is not None:
            self._dataset_mean = mean

    def is_fitted(self) -> bool:
        return self._dataset_mean is not None


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

    def _legacy_entity_means_key(self) -> str:
        raise NotImplementedError

    def get_state(self) -> dict[str, object]:
        if self._dataset_mean is None:
            return {}
        return {
            "dataset_mean": self._dataset_mean,
            self._legacy_entity_means_key(): dict(self._entity_means),
        }

    def set_state(self, state: dict[str, object]) -> None:
        mean = state_float(state, "dataset_mean")
        if mean is not None:
            self._dataset_mean = mean
        entity_means = state_str_dict(state, self._legacy_entity_means_key())
        if entity_means:
            self._entity_means = entity_means

    def is_fitted(self) -> bool:
        return self._dataset_mean is not None


@register_predictor(
    "naiveDrugMean",
    description="Predict per-drug mean response with global fallback.",
    category="baseline",
)
class NaiveDrugMeanPredictor(_SingleEntityNaivePredictor):
    """Naive drug mean predictor component."""

    def _entity_keys(self, pair_context: PairContext) -> np.ndarray:
        return pair_context.drug_ids

    def _legacy_entity_means_key(self) -> str:
        return "drug_means"


@register_predictor(
    "naiveCellLineMean",
    description="Predict per-cell-line mean response with global fallback.",
    category="baseline",
)
class NaiveCellLineMeanPredictor(_SingleEntityNaivePredictor):
    """Naive cell line mean predictor component."""

    def _entity_keys(self, pair_context: PairContext) -> np.ndarray:
        return pair_context.cell_line_ids

    def _legacy_entity_means_key(self) -> str:
        return "cell_line_means"


@register_predictor(
    "naiveTissueMean",
    description="Predict per-tissue mean response with global fallback.",
    category="baseline",
)
class NaiveTissueMeanPredictor(_SingleEntityNaivePredictor):
    """Naive tissue mean predictor component."""

    def _entity_keys(self, pair_context: PairContext) -> np.ndarray:
        if pair_context.tissue_ids is None:
            msg = "NaiveTissueMeanPredictor requires tissue_ids in pair_context"
            raise ValueError(msg)
        return pair_context.tissue_ids

    def _legacy_entity_means_key(self) -> str:
        return "tissue_means"


@register_predictor(
    "naiveTissueDrugMean",
    description="Predict per tissue-drug combination mean response.",
    category="baseline",
)
class NaiveTissueDrugMeanPredictor(BaselinePredictor):
    """Naive tissue drug mean predictor component."""

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


@register_predictor(
    "naiveMeanEffects",
    description="Predict mean plus cell-line and drug effects.",
    category="baseline",
)
class NaiveMeanEffectsPredictor(BaselinePredictor):
    """Naive mean effects predictor component."""

    def __init__(self) -> None:
        self._dataset_mean: float | None = None
        self._tissue_effects: dict[str, float] = {}
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
        cell_line_ids = pair_context.cell_line_ids.astype(str)
        drug_ids = pair_context.drug_ids.astype(str)

        cell_line_means: dict[str, float] = {}
        for cell_id in np.unique(cell_line_ids):
            mask = cell_line_ids == cell_id
            cell_line_means[str(cell_id)] = float(np.mean(y[mask]))

        if pair_context.tissue_ids is not None:
            tissues = np.asarray(pair_context.tissue_ids).reshape(-1)
            tissue_means: dict[str, float] = {}
            for tissue in np.unique(tissues):
                tissue_key = str(tissue.item() if isinstance(tissue, np.ndarray) else tissue)
                mask = np.array([str(t.item() if isinstance(t, np.ndarray) else t) for t in tissues]) == tissue_key
                if np.any(mask):
                    tissue_means[tissue_key] = float(np.mean(y[mask]))

            self._tissue_effects = {tissue: mean - self._dataset_mean for tissue, mean in tissue_means.items()}

            cell_line_to_tissue: dict[str, str] = {}
            for cell_id in np.unique(cell_line_ids):
                mask = cell_line_ids == cell_id
                tissue = tissues[mask][0]
                tissue_key = str(tissue.item() if isinstance(tissue, np.ndarray) else tissue)
                cell_line_to_tissue[str(cell_id)] = tissue_key

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
        if self._tissue_effects and pair_context.tissue_ids is not None:
            tissues = np.asarray(pair_context.tissue_ids).reshape(-1)
            return np.array(
                [
                    self._dataset_mean
                    + self._tissue_effects.get(
                        str(tissue.item() if isinstance(tissue, np.ndarray) else tissue),
                        0.0,
                    )
                    + self._cell_line_effects.get(str(cell_id), 0.0)
                    + self._drug_effects.get(str(drug_id), 0.0)
                    for cell_id, drug_id, tissue in zip(
                        pair_context.cell_line_ids,
                        pair_context.drug_ids,
                        tissues,
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
                for cell_id, drug_id in zip(
                    pair_context.cell_line_ids,
                    pair_context.drug_ids,
                    strict=True,
                )
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
