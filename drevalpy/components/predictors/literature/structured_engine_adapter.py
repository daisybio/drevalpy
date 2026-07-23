"""Shared structured-literature engine adapter."""

from __future__ import annotations

import importlib
import io
from typing import Any, ClassVar

import joblib
import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.literature._engine_base import LiteratureEngineBase
from drevalpy.components.predictors.literature._feature_dataset_from_batch import (
    feature_dataset_from_blocks,
    merge_feature_dataset,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.predictors.structured import StructuredPredictor
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.config import PredictionMode

ENGINE_MODULES: dict[str, str] = {
    "PrecilyModel": "drevalpy.components.predictors.literature.impl.precily.precily",
    "SRMF": "drevalpy.components.predictors.literature.impl.srmf.srmf",
    "MOLIR": "drevalpy.components.predictors.literature.impl.molir.molir",
    "SuperFELTR": "drevalpy.components.predictors.literature.impl.superfeltr.superfeltr",
    "PharmaFormerModel": "drevalpy.components.predictors.literature.impl.pharmaformer.pharmaformer",
    "DIPKModel": "drevalpy.components.predictors.literature.impl.dipk.dipk",
    "SparseGOModel": "drevalpy.components.predictors.literature.impl.sparsego.sparsego",
}


def resolve_engine_cls(class_name: str) -> type[LiteratureEngineBase]:
    """Import a literature engine class by its legacy class name."""
    module_path = ENGINE_MODULES.get(class_name)
    if module_path is None:
        msg = f"Unknown literature engine class: {class_name}"
        raise ValueError(msg)
    module = importlib.import_module(module_path)
    engine_cls = getattr(module, class_name, None)
    if engine_cls is None or not issubclass(engine_cls, LiteratureEngineBase):
        msg = f"Module {module_path!r} does not export {class_name}"
        raise ValueError(msg)
    return engine_cls


def _rebind_engine_class(engine: LiteratureEngineBase, class_name: str) -> LiteratureEngineBase:
    """Ensure *engine*'s class object matches the currently imported module.

    Coverage and lazy re-imports can leave instances bound to a stale class
    object that pickle rejects with ``not the same object as ...``.
    """
    current_cls = resolve_engine_cls(class_name)
    if type(engine) is not current_cls and type(engine).__name__ == current_cls.__name__:
        engine.__class__ = current_cls
    return engine


class StructuredLiteratureEnginePredictor(StructuredPredictor):
    """Train a component-owned literature engine on featurizer-produced blocks."""

    _engine_class_name: ClassVar[str]
    requires_raw_feature_datasets: ClassVar[bool] = False
    requires_drug_featurizer: ClassVar[bool] = True
    supports_early_stopping: ClassVar[bool] = False
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    _ENGINE_PRELOAD_ATTRS: ClassVar[tuple[str, ...]] = (
        "layer_connections",
        "gene2id_mapping_ont",
        "ontology_gene_order",
        "gene_dim_input",
        "model",
    )

    def __init__(self) -> None:
        self._hyperparameters: dict[str, Any] = {}
        self._engine: LiteratureEngineBase | None = None
        self._engine_preload_state: dict[str, Any] = {}

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

    def build(self, hyperparameters: dict[str, Any], input_dims: dict[str, Any]) -> None:
        _ = input_dims
        self._hyperparameters = dict(hyperparameters)
        self._engine = None

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
    ) -> FeatureDataset | None:
        _ = model_name
        engine = cls.engine_cls()()
        if hyperparameters:
            engine.hyperparameters = dict(hyperparameters)
        features = engine.load_drug_features(data_path, dataset_name)
        if hyperparameters is not None:
            hyperparameters.update(dict(engine.hyperparameters))
        return features

    def _materialize_inputs(
        self,
        batch: ModelInputBatch,
    ) -> tuple[FeatureDataset, FeatureDataset | None]:
        cell_line_input = batch.cell_line_input
        drug_input = batch.drug_input
        if cell_line_input is None:
            msg = "structured literature predictor requires cell_line_input"
            raise RuntimeError(msg)
        cell_lines = cell_line_input
        if batch.cell_line_blocks:
            cell_lines = merge_feature_dataset(cell_line_input, batch.cell_line_blocks, batch.cell_line_entity_ids)
        if not self.requires_drug_featurizer:
            return cell_lines, None
        if drug_input is None:
            msg = "structured literature predictor requires drug_input"
            raise RuntimeError(msg)
        if batch.drug_blocks and batch.drug_entity_ids is not None:
            drugs = merge_feature_dataset(drug_input, batch.drug_blocks, batch.drug_entity_ids)
        elif batch.drug_entity_ids is not None:
            drugs = feature_dataset_from_blocks(batch.drug_entity_ids, batch.drug_blocks, fallback=drug_input)
        else:
            drugs = drug_input
        return cell_lines, drugs

    def fit(self, batch: ModelInputBatch) -> None:
        if batch.response is None:
            msg = "structured literature predictor requires response"
            raise RuntimeError(msg)
        if batch.cell_line_features.size == 0 and not self.requires_raw_feature_datasets:
            msg = "cell_line featurizer produced no features"
            raise ValueError(msg)
        cell_line_input = batch.cell_line_input
        drug_input = batch.drug_input
        if self.requires_raw_feature_datasets:
            if cell_line_input is None:
                msg = "structured literature predictor requires cell_line_input"
                raise RuntimeError(msg)
            cell_lines = cell_line_input
            drugs = None if not self.requires_drug_featurizer else drug_input
        else:
            cell_lines, drugs = self._materialize_inputs(batch)
        output = DrugResponseDataset(
            response=batch.response,
            cell_line_ids=batch.cell_line_ids,
            drug_ids=batch.drug_ids,
        )
        hyperparameters = dict(self._hyperparameters)
        engine = self.engine_cls()()
        for name, value in self._engine_preload_state.items():
            setattr(engine, name, value)
        engine.build_model(hyperparameters)
        engine.train(
            output,
            cell_lines,
            drugs,
            output_earlystopping=batch.early_stopping_response,
            model_checkpoint_dir=batch.training_context.checkpoint_dir,
        )
        self._engine = engine

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        cell_line_input = batch.cell_line_input
        if self._engine is None or cell_line_input is None:
            return np.full(batch.n_pairs, np.nan, dtype=np.float64)
        drug_input = batch.drug_input
        if self.requires_raw_feature_datasets:
            drugs = None if not self.requires_drug_featurizer else drug_input
            return self._engine.predict(
                batch.cell_line_ids,
                batch.drug_ids,
                cell_line_input,
                drugs,
            )
        cell_lines, drugs = self._materialize_inputs(batch)
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
        engine = _rebind_engine_class(self._engine, self._engine_class_name)
        self._engine = engine
        # Keep the class-level cache aligned with the rebound class.
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
