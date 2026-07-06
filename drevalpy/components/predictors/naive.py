"""Naive baseline predictors."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.pair_batch import PairBatch
from drevalpy.components.predictors._identity_batch import pair_tissue_ids
from drevalpy.components.predictors.baseline import BaselinePredictor
from drevalpy.components.predictors.structured import StructuredPredictor
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
    ) -> None:
        _ = x
        self._dataset_mean = float(np.mean(y))

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        if self._dataset_mean is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        return np.full(len(x), self._dataset_mean, dtype=np.float64)

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


class _SingleEntityNaivePredictor(StructuredPredictor):

    def __init__(self) -> None:
        self._dataset_mean: float | None = None
        self._entity_means: dict[str, float] = {}

    def _entity_keys(self, batch: PairBatch) -> np.ndarray:
        raise NotImplementedError

    def fit_structured(
        self,
        batch: PairBatch,
        *,
        output=None,
        cell_line_input=None,
        drug_input=None,
        output_earlystopping=None,
    ) -> None:
        _ = output, cell_line_input, drug_input, output_earlystopping
        if batch.response is None:
            msg = "Naive predictors require response values during fit"
            raise ValueError(msg)
        y = np.asarray(batch.response, dtype=np.float64)
        self._dataset_mean = float(np.mean(y))
        keys = self._entity_keys(batch)
        for entity in np.unique(keys.astype(str)):
            mask = keys.astype(str) == entity
            self._entity_means[str(entity)] = float(np.mean(y[mask]))

    def predict_structured(
        self,
        batch: PairBatch,
        *,
        cell_line_input=None,
        drug_input=None,
    ) -> np.ndarray:
        _ = cell_line_input, drug_input
        if self._dataset_mean is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        keys = self._entity_keys(batch).astype(str)
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
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
)
class NaiveDrugMeanPredictor(_SingleEntityNaivePredictor):
    """Naive drug mean predictor component."""

    def _entity_keys(self, batch: PairBatch) -> np.ndarray:
        return batch.drug_ids

    def _legacy_entity_means_key(self) -> str:
        return "drug_means"


@register_predictor(
    "naiveCellLineMean",
    description="Predict per-cell-line mean response with global fallback.",
    category="baseline",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
)
class NaiveCellLineMeanPredictor(_SingleEntityNaivePredictor):
    """Naive cell line mean predictor component."""

    def _entity_keys(self, batch: PairBatch) -> np.ndarray:
        return batch.cell_line_ids

    def _legacy_entity_means_key(self) -> str:
        return "cell_line_means"


@register_predictor(
    "naiveTissueMean",
    description="Predict per-tissue mean response with global fallback.",
    category="baseline",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
)
class NaiveTissueMeanPredictor(StructuredPredictor):
    """Naive tissue mean predictor component."""


    def __init__(self) -> None:
        self._dataset_mean: float | None = None
        self._entity_means: dict[str, float] = {}

    def fit_structured(
        self,
        batch: PairBatch,
        *,
        output=None,
        cell_line_input=None,
        drug_input=None,
        output_earlystopping=None,
    ) -> None:
        _ = output, cell_line_input, drug_input, output_earlystopping
        if batch.response is None:
            msg = "Naive predictors require response values during fit"
            raise ValueError(msg)
        tissue_ids = pair_tissue_ids(batch, cell_line_input=cell_line_input)
        if tissue_ids is None:
            msg = "NaiveTissueMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        y = np.asarray(batch.response, dtype=np.float64)
        self._dataset_mean = float(np.mean(y))
        for tissue in np.unique(tissue_ids.astype(str)):
            mask = tissue_ids.astype(str) == tissue
            self._entity_means[str(tissue)] = float(np.mean(y[mask]))

    def predict_structured(
        self,
        batch: PairBatch,
        *,
        cell_line_input=None,
        drug_input=None,
    ) -> np.ndarray:
        _ = cell_line_input, drug_input
        if self._dataset_mean is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        tissue_ids = pair_tissue_ids(batch, cell_line_input=cell_line_input)
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
class NaiveTissueDrugMeanPredictor(StructuredPredictor):
    """Naive tissue drug mean predictor component."""


    def __init__(self) -> None:
        self._dataset_mean: float | None = None
        self._combo_means: dict[str, float] = {}

    def fit_structured(
        self,
        batch: PairBatch,
        *,
        output=None,
        cell_line_input=None,
        drug_input=None,
        output_earlystopping=None,
    ) -> None:
        _ = output, cell_line_input, drug_input, output_earlystopping
        if batch.response is None:
            msg = "Naive predictors require response values during fit"
            raise ValueError(msg)
        tissue_ids = pair_tissue_ids(batch, cell_line_input=cell_line_input)
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

    def predict_structured(
        self,
        batch: PairBatch,
        *,
        cell_line_input=None,
        drug_input=None,
    ) -> np.ndarray:
        _ = cell_line_input, drug_input
        if self._dataset_mean is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        tissue_ids = pair_tissue_ids(batch, cell_line_input=cell_line_input)
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


@register_predictor(
    "naiveMeanEffects",
    description="Predict mean plus cell-line and drug effects.",
    category="baseline",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
)
class NaiveMeanEffectsPredictor(StructuredPredictor):
    """Naive mean effects predictor component."""


    def __init__(self) -> None:
        self._dataset_mean: float | None = None
        self._tissue_effects: dict[str, float] = {}
        self._cell_line_effects: dict[str, float] = {}
        self._drug_effects: dict[str, float] = {}

    def fit_structured(
        self,
        batch: PairBatch,
        *,
        output=None,
        cell_line_input=None,
        drug_input=None,
        output_earlystopping=None,
    ) -> None:
        _ = output, cell_line_input, drug_input, output_earlystopping
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

        tissue_ids = pair_tissue_ids(batch, cell_line_input=cell_line_input)
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

    def predict_structured(
        self,
        batch: PairBatch,
        *,
        cell_line_input=None,
        drug_input=None,
    ) -> np.ndarray:
        _ = cell_line_input, drug_input
        if self._dataset_mean is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        tissue_ids = pair_tissue_ids(batch, cell_line_input=cell_line_input)
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
