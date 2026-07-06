"""Zoo-backed sklearn baseline adapters without bespoke model implementations."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.data_loading import (
    load_cell_line_features_for_model_config,
    load_drug_features_for_model_config,
)
from drevalpy.components.register_builtins import ensure_components_registered
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.models._component_bridge import preview_sklearn_estimator
from drevalpy.models.config import ModelConfig
from drevalpy.models.featurizer_mapping import cell_line_views_from_model_config, drug_views_from_model_config
from drevalpy.models.zoo import zoo_model_config

from .sklearn_models import SklearnModel


class ZooPresetSklearnModel(SklearnModel):
    """Sklearn baseline whose featurizers and predictor come entirely from a zoo entry."""

    _zoo_name: ClassVar[str]

    @classmethod
    def get_model_name(cls) -> str:
        return cls._zoo_name

    def _zoo_config(self) -> ModelConfig:
        return zoo_model_config(self._zoo_name, self.hyperparameters)

    def build_model(self, hyperparameters: dict) -> None:
        if self._zoo_name == "MultiViewXGBoost":
            try:
                import xgboost  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "MultiViewXGBoost requires the optional 'xgboost' extra. "
                    "Install it with: pip install drevalpy[xgboost] (or `poetry install -E xgboost`)."
                ) from exc
        config = self._zoo_config()
        self.cell_line_views = cell_line_views_from_model_config(config)
        self.drug_views = drug_views_from_model_config(config)
        merged = dict(hyperparameters)
        merged.setdefault("cell_line_views", self.cell_line_views)
        merged.setdefault("drug_views", self.drug_views)
        super().build_model(merged)

    def _init_component_model(self) -> None:
        ensure_components_registered()
        config = self._zoo_config()
        self._component_bridge.set_composed_config(config)
        self._preview_model = preview_sklearn_estimator(self._component_bridge, self.hyperparameters)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        config = self._zoo_config()
        return load_cell_line_features_for_model_config(
            config,
            data_path,
            dataset_name,
            model_name=self.get_model_name(),
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        config = self._zoo_config()
        return load_drug_features_for_model_config(
            config,
            data_path,
            dataset_name,
            model_name=self.get_model_name(),
        )


class MultiViewRandomForest(ZooPresetSklearnModel):
    """Multi-view random forest resolved from the zoo preset."""

    _zoo_name = "MultiViewRandomForest"


class MultiViewXGBoost(ZooPresetSklearnModel):
    """Multi-view XGBoost resolved from the zoo preset."""

    _zoo_name = "MultiViewXGBoost"


class MultiViewLightGBM(ZooPresetSklearnModel):
    """Multi-view LightGBM resolved from the zoo preset."""

    _zoo_name = "MultiViewLightGBM"

    def build_model(self, hyperparameters: dict) -> None:
        try:
            import lightgbm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MultiViewLightGBM requires the optional 'lightgbm' extra. "
                "Install it with: pip install drevalpy[lightgbm] (or `poetry install -E lightgbm`)."
            ) from exc
        super().build_model(hyperparameters)
