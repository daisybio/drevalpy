"""Tests for structured literature engine adapter."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np

from drevalpy.components.predictors.literature._engine_base import LiteratureEngineBase
from drevalpy.components.predictors.literature.structured_engine_adapter import (
    DISCOVERED_HYPERPARAMETERS_KEY,
    ENGINE_MODULES,
    StructuredLiteratureEnginePredictor,
    resolve_engine_cls,
)
from drevalpy.datasets.dataset import FeatureDataset


def test_engine_modules_map_to_impl_packages() -> None:
    assert "PrecilyModel" in ENGINE_MODULES
    assert all("literature.impl" in path for path in ENGINE_MODULES.values())


def test_structured_engine_adapter_avoids_models_package_imports() -> None:
    module = importlib.import_module("drevalpy.components.predictors.literature.structured_engine_adapter")
    source_path = module.__file__
    assert source_path is not None
    text = Path(source_path).read_text(encoding="utf-8")
    assert "drevalpy.models.DIPK" not in text
    assert "drevalpy.components.predictors.literature.impl" in text


def test_resolve_engine_cls_imports_srmf_engine() -> None:
    from drevalpy.components.predictors.literature._engine_base import LiteratureEngineBase

    engine_cls = resolve_engine_cls("SRMF")
    assert issubclass(engine_cls, LiteratureEngineBase)


class _DiscoveringEngine:
    def __init__(self) -> None:
        self.hyperparameters: dict[str, Any] = {}

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        return {}

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        _ = data_path, dataset_name
        self.hyperparameters["drug_dim"] = 128
        return FeatureDataset(features={"d1": {"fingerprints": np.array([1.0, 0.0])}})


class _DiscoveringPredictor(StructuredLiteratureEnginePredictor):
    _engine_class_name: ClassVar[str] = "DiscoveringEngine"

    @classmethod
    def engine_cls(cls) -> type[LiteratureEngineBase]:
        return cast(type[LiteratureEngineBase], _DiscoveringEngine)


def test_load_dataset_drug_features_returns_discovered_hps_without_mutating_caller() -> None:
    hyperparameters = {"epochs": 1, "drug_dim": 2048}
    features, preload = _DiscoveringPredictor.load_dataset_drug_features(
        ".",
        "TOY",
        hyperparameters=hyperparameters,
    )
    assert hyperparameters == {"epochs": 1, "drug_dim": 2048}
    assert features is not None
    assert preload[DISCOVERED_HYPERPARAMETERS_KEY] == {"drug_dim": 128}


def test_fit_merges_discovered_hyperparameters_into_configure() -> None:
    configured: dict[str, Any] = {}

    class _ConfigurableEngine(_DiscoveringEngine):
        def configure(self, hyperparameters: dict[str, Any]) -> None:
            configured.clear()
            configured.update(hyperparameters)

        def train(self, *args: Any, **kwargs: Any) -> None:
            _ = args, kwargs

    class _ConfigurablePredictor(_DiscoveringPredictor):
        @classmethod
        def engine_cls(cls) -> type[LiteratureEngineBase]:
            return cast(type[LiteratureEngineBase], _ConfigurableEngine)

    predictor = _ConfigurablePredictor(hyperparameters={"epochs": 1, "drug_dim": 2048})
    predictor.set_engine_preload_state({DISCOVERED_HYPERPARAMETERS_KEY: {"drug_dim": 128}})

    from drevalpy.components.model_input_batch import ModelInputBatch
    from drevalpy.components.training_context import TrainingContext
    from drevalpy.datasets.dataset import DrugResponseDataset

    cell_line_input = FeatureDataset(features={"cl1": {"gene_expression": np.array([0.1, 0.2])}})
    drug_input = FeatureDataset(features={"d1": {"fingerprints": np.array([1.0, 0.0])}})
    response = DrugResponseDataset(
        response=np.array([1.0]),
        cell_line_ids=np.array(["cl1"]),
        drug_ids=np.array(["d1"]),
    )
    batch = ModelInputBatch.from_response(
        response,
        cell_line_entity_ids=np.array(["cl1"]),
        drug_entity_ids=np.array(["d1"]),
        cell_line_features=np.array([[0.1, 0.2]]),
        drug_features=np.array([[1.0, 0.0]]),
        cell_line_pair_idx=np.array([0]),
        drug_pair_idx=np.array([0]),
        cell_line_input=cell_line_input,
        drug_input=drug_input,
        training_context=TrainingContext(checkpoint_dir="checkpoints"),
    )
    predictor.fit(batch)
    assert configured["drug_dim"] == 128
    assert configured["epochs"] == 1
