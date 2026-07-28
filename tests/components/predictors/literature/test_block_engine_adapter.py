"""Tests for block literature engine adapter."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import numpy as np
import pytest

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.literature._engine_base import LiteratureEngineBase
from drevalpy.components.predictors.literature.block_engine_adapter import BlockLiteratureEnginePredictor
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


class _BlockPredictor(BlockLiteratureEnginePredictor):
    _engine_class_name: ClassVar[str] = "RecordingEngine"
    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("pathways",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("smilesvec",)

    @classmethod
    def engine_cls(cls) -> type[LiteratureEngineBase]:
        return cast(type[LiteratureEngineBase], _RecordingEngine)


def _batch(*, drug_blocks: dict[str, np.ndarray] | None = None) -> ModelInputBatch:
    response = DrugResponseDataset(
        response=np.array([1.0]),
        cell_line_ids=np.array(["cl1"]),
        drug_ids=np.array(["d1"]),
    )
    return ModelInputBatch.from_response(
        response,
        cell_line_entity_ids=np.array(["cl1"]),
        drug_entity_ids=np.array(["d1"]),
        cell_line_features=np.array([[0.1]]),
        drug_features=np.array([[0.2]]),
        cell_line_pair_idx=np.array([0]),
        drug_pair_idx=np.array([0]),
        cell_line_blocks={"pathways": np.array([[0.1, 0.2]])},
        drug_blocks=drug_blocks if drug_blocks is not None else {"smilesvec": np.array([[0.3, 0.4]])},
        cell_line_input=FeatureDataset(features={"cl1": {"gene_expression": np.array([9.0])}}),
        drug_input=FeatureDataset(features={"d1": {"fingerprints": np.array([8.0])}}),
        training_context=TrainingContext(checkpoint_dir="checkpoints"),
    )


def test_block_adapter_uses_declared_blocks_not_raw_fallback() -> None:
    predictor = _BlockPredictor(hyperparameters={})
    predictor.fit(_batch())
    engine = cast(_RecordingEngine, predictor._engine)
    assert engine.seen_cell is not None
    views = next(iter(engine.seen_cell.features.values()))
    assert "pathways" in views
    assert "gene_expression" not in views
    assert engine.seen_drug is not None
    drug_views = next(iter(engine.seen_drug.features.values()))
    assert "smilesvec" in drug_views
    assert "fingerprints" not in drug_views


def test_block_adapter_rejects_missing_block() -> None:
    predictor = _BlockPredictor(hyperparameters={})
    with pytest.raises(ValueError, match="missing drug block"):
        predictor.fit(_batch(drug_blocks={}))
