"""Tests for raw literature engine adapter."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import numpy as np
import pytest

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.literature._engine_base import LiteratureEngineBase
from drevalpy.components.predictors.literature.raw_engine_adapter import RawLiteratureEnginePredictor
from drevalpy.components.training_context import TrainingContext
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset


class _RecordingEngine:
    def __init__(self) -> None:
        self.seen_cell: FeatureDataset | None = None
        self.seen_drug: FeatureDataset | None = None
        self.hyperparameters: dict[str, Any] = {}

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        return {}

    def configure(self, hyperparameters: dict[str, Any]) -> None:
        self.hyperparameters = dict(hyperparameters)

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        _ = output, args, kwargs
        self.seen_cell = cell_line_input
        self.seen_drug = drug_input

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray:
        _ = args, kwargs
        return np.array([0.5])


class _RawPredictor(RawLiteratureEnginePredictor):
    _engine_class_name: ClassVar[str] = "RecordingEngine"
    required_cell_line_views: ClassVar[tuple[str, ...]] = ("gene_expression",)
    required_drug_views: ClassVar[tuple[str, ...]] = ("molgnet_features",)

    @classmethod
    def engine_cls(cls) -> type[LiteratureEngineBase]:
        return cast(type[LiteratureEngineBase], _RecordingEngine)


def _batch(*, cell_views: dict[str, np.ndarray] | None = None) -> ModelInputBatch:
    response = DrugResponseDataset(
        response=np.array([1.0]),
        cell_line_ids=np.array(["cl1"]),
        drug_ids=np.array(["d1"]),
    )
    cell_features = {"cl1": cell_views or {"gene_expression": np.array([1.0, 2.0])}}
    return ModelInputBatch.from_response(
        response,
        cell_line_entity_ids=np.array(["cl1"]),
        drug_entity_ids=np.array(["d1"]),
        cell_line_features=np.array([[0.0]]),
        drug_features=None,
        cell_line_pair_idx=np.array([0]),
        drug_pair_idx=np.array([0]),
        cell_line_input=FeatureDataset(features=cell_features),
        drug_input=FeatureDataset(features={"d1": {"molgnet_features": np.array([0.1])}}),
        training_context=TrainingContext(checkpoint_dir="checkpoints"),
    )


def test_raw_adapter_passes_feature_datasets() -> None:
    predictor = _RawPredictor(hyperparameters={})
    predictor.fit(_batch())
    engine = cast(_RecordingEngine, predictor._engine)
    assert engine.seen_cell is not None
    assert "gene_expression" in next(iter(engine.seen_cell.features.values()))
    assert engine.seen_drug is not None
    assert "molgnet_features" in next(iter(engine.seen_drug.features.values()))


def test_raw_adapter_rejects_missing_view() -> None:
    predictor = _RawPredictor(hyperparameters={})
    with pytest.raises(ValueError, match="missing cell_line view"):
        predictor.fit(_batch(cell_views={"mutations": np.array([1.0])}))
