"""Tests for shared literature engine mixin helpers."""

from __future__ import annotations

from typing import Any, ClassVar, cast

from drevalpy.components.predictors.literature._engine_base import LiteratureEngineBase
from drevalpy.components.predictors.literature._engine_mixin import LiteratureEngineMixin
from drevalpy.components.predictors.literature._engine_resolve import DISCOVERED_HYPERPARAMETERS_KEY
from drevalpy.components.predictors.raw_dataset import RawDatasetPredictor
from drevalpy.datasets.dataset import FeatureDataset


class _DiscoveringEngine:
    def __init__(self) -> None:
        self.hyperparameters: dict[str, Any] = {}

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        return {}

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        _ = data_path, dataset_name
        self.hyperparameters["drug_dim"] = 128
        return FeatureDataset(features={"d1": {"fingerprints": __import__("numpy").array([1.0, 0.0])}})


class _MixinPredictor(LiteratureEngineMixin, RawDatasetPredictor):
    _engine_class_name: ClassVar[str] = "DiscoveringEngine"

    @classmethod
    def engine_cls(cls) -> type[LiteratureEngineBase]:
        return cast(type[LiteratureEngineBase], _DiscoveringEngine)


def test_load_dataset_drug_features_returns_discovered_hps_without_mutating_caller() -> None:
    hyperparameters = {"epochs": 1, "drug_dim": 2048}
    features, preload = _MixinPredictor.load_dataset_drug_features(
        ".",
        "TOY",
        hyperparameters=hyperparameters,
    )
    assert hyperparameters == {"epochs": 1, "drug_dim": 2048}
    assert features is not None
    assert preload[DISCOVERED_HYPERPARAMETERS_KEY] == {"drug_dim": 128}
