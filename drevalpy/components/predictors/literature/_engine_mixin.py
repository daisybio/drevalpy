"""Shared literature-engine lifecycle helpers."""

from __future__ import annotations

import io
from typing import Any, ClassVar

import joblib
import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.base import Predictor
from drevalpy.components.predictors.literature._engine_base import LiteratureEngineBase
from drevalpy.components.predictors.literature._engine_resolve import (
    DISCOVERED_HYPERPARAMETERS_KEY,
    rebind_engine_class,
    resolve_engine_cls,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.config import PredictionMode


class LiteratureEngineMixin:
    """Lifecycle helpers shared by block and raw literature predictors."""

    _engine_class_name: ClassVar[str]
    supports_early_stopping: ClassVar[bool] = False
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})
    _ENGINE_PRELOAD_ATTRS: ClassVar[tuple[str, ...]] = (
        "layer_connections",
        "gene2id_mapping_ont",
        "ontology_gene_order",
        "gene_dim_input",
        "model",
    )

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        Predictor.__init__(self, hyperparameters)  # type: ignore[arg-type]
        self._engine: LiteratureEngineBase | None = None
        self._engine_preload_state: dict[str, Any] = {}
        self._hyperparameters: dict[str, Any]

    @classmethod
    def engine_cls(cls) -> type[LiteratureEngineBase]:
        """Lazily import the parity-checked literature engine class."""
        cache_attr = "_cached_engine_cls"
        cached = getattr(cls, cache_attr, None)
        if cached is None:
            cached = resolve_engine_cls(cls._engine_class_name)
            setattr(cls, cache_attr, cached)
        return cached

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        return dict(cls.engine_cls().get_default_hyperparameters())

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        engine_cls = cls.engine_cls()
        engine_space = getattr(engine_cls, "get_hyperparameter_space", None)
        if callable(engine_space):
            return dict(engine_space())
        return {}

    def set_engine_preload_state(self, state: dict[str, Any]) -> None:
        self._engine_preload_state = dict(state)

    @classmethod
    def load_dataset_cell_line_features(
        cls,
        data_path: str,
        dataset_name: str,
        *,
        hyperparameters: dict[str, Any] | None = None,
        model_name: str | None = None,
    ) -> tuple[FeatureDataset, dict[str, Any]]:
        _ = model_name
        engine = cls.engine_cls()()
        if hyperparameters:
            engine.hyperparameters = dict(hyperparameters)
        features = engine.load_cell_line_features(data_path, dataset_name)
        preload = {
            attr: getattr(engine, attr) for attr in cls._ENGINE_PRELOAD_ATTRS if getattr(engine, attr, None) is not None
        }
        return features, preload

    @classmethod
    def load_dataset_drug_features(
        cls,
        data_path: str,
        dataset_name: str,
        *,
        hyperparameters: dict[str, Any] | None = None,
        model_name: str | None = None,
    ) -> tuple[FeatureDataset | None, dict[str, Any]]:
        _ = model_name
        engine = cls.engine_cls()()
        seed_hyperparameters = dict(hyperparameters) if hyperparameters else {}
        if seed_hyperparameters:
            engine.hyperparameters = dict(seed_hyperparameters)
        features = engine.load_drug_features(data_path, dataset_name)
        discovered = {
            key: value
            for key, value in dict(engine.hyperparameters).items()
            if key not in seed_hyperparameters or seed_hyperparameters[key] != value
        }
        preload: dict[str, Any] = {}
        if discovered:
            preload[DISCOVERED_HYPERPARAMETERS_KEY] = discovered
        return features, preload

    def _train_engine(
        self,
        batch: ModelInputBatch,
        cell_lines: FeatureDataset,
        drugs: FeatureDataset | None,
    ) -> None:
        if batch.response is None:
            msg = "literature predictor requires response"
            raise RuntimeError(msg)
        output = DrugResponseDataset(
            response=batch.response,
            cell_line_ids=batch.cell_line_ids,
            drug_ids=batch.drug_ids,
        )
        hyperparameters = dict(self._hyperparameters)
        preload = dict(self._engine_preload_state)
        discovered = preload.pop(DISCOVERED_HYPERPARAMETERS_KEY, None)
        if isinstance(discovered, dict):
            hyperparameters.update(discovered)
        engine = self.engine_cls()()
        for name, value in preload.items():
            setattr(engine, name, value)
        engine.configure(hyperparameters)
        engine.train(
            output,
            cell_lines,
            drugs,
            output_earlystopping=batch.early_stopping_response,
            model_checkpoint_dir=batch.training_context.checkpoint_dir,
        )
        self._engine = engine

    def _predict_engine(
        self,
        batch: ModelInputBatch,
        cell_lines: FeatureDataset,
        drugs: FeatureDataset | None,
    ) -> np.ndarray:
        if self._engine is None:
            return np.full(batch.n_pairs, np.nan, dtype=np.float64)
        return self._engine.predict(
            batch.cell_line_ids,
            batch.drug_ids,
            cell_lines,
            drugs,
        )

    def is_fitted(self) -> bool:
        return self._engine is not None

    def get_state(self) -> dict[str, object]:
        if self._engine is None:
            return {}
        engine = rebind_engine_class(self._engine, self._engine_class_name)
        self._engine = engine
        cache_attr = "_cached_engine_cls"
        setattr(type(self), cache_attr, type(engine))
        buffer = io.BytesIO()
        joblib.dump(engine, buffer)
        return {
            "hyperparameters": dict(self._hyperparameters),
            "engine": buffer.getvalue(),
        }

    def set_state(self, state: dict[str, object]) -> None:
        engine_blob = state.get("engine")
        if not isinstance(engine_blob, (bytes, bytearray)):
            msg = f"{self.__class__.__name__} state requires an engine byte blob"
            raise PredictorStateError(msg)
        try:
            loaded = joblib.load(io.BytesIO(engine_blob))
        except Exception as exc:
            msg = f"{self.__class__.__name__} engine blob could not be deserialized"
            raise PredictorStateError(msg) from exc
        if not isinstance(loaded, LiteratureEngineBase):
            msg = f"{self.__class__.__name__} engine blob did not deserialize to a literature engine"
            raise PredictorStateError(msg)
        hyperparameters = state.get("hyperparameters")
        if not isinstance(hyperparameters, dict):
            msg = f"{self.__class__.__name__} state is missing hyperparameters"
            raise PredictorStateError(msg)
        self._engine = loaded
        self._hyperparameters = dict(hyperparameters)
