"""Public API for constructing DRPModel classes from modular spec strings."""

from __future__ import annotations

import json
import os
from typing import Any

from drevalpy.components.data_loading import load_cell_line_feature_views, load_drug_feature_views
from drevalpy.components.register_builtins import ensure_components_registered
import numpy as np

from drevalpy.components.tuning.drp_hyperparameters import (
    config_from_build_hyperparameters,
    default_hyperparameters_for_drp_model,
    flat_hyperparameters_from_model_config,
    structured_space_for_drp_model,
)
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models._component_bridge import (
    ComponentDRPBridge,
    load_component_stack,
    save_component_stack,
)
from drevalpy.models.config import ModelConfig
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.featurizer_mapping import cell_line_views_from_model_config, drug_views_from_model_config


def construct_model(name: str, spec: str) -> type[DRPModel]:
    """Return a DRPModel subclass for *spec* with ``get_model_name() == name``.

    The returned class uses the existing ``ModelConfig`` / ``ComposedModel`` stack via
    ``ComponentDRPBridge`` and does not duplicate composition logic.
    """
    ensure_components_registered()
    base_config = ModelConfig.from_spec(spec)
    base_config.validate()

    class ConstructedDRPModel(DRPModel):
        _model_spec: str = spec
        _display_name: str = name
        _base_config: ModelConfig = base_config

        def __init__(self) -> None:
            super().__init__()
            self._bridge = ComponentDRPBridge()
            self.hyperparameters: dict[str, Any] = {}
            self._cell_line_views_list: list[str] = cell_line_views_from_model_config(base_config)
            self._drug_views_list: list[str] = drug_views_from_model_config(base_config)

        @classmethod
        def get_model_name(cls) -> str:
            return name

        @classmethod
        def get_default_hyperparameters(cls) -> dict[str, Any]:
            return default_hyperparameters_for_drp_model(cls)

        @classmethod
        def get_structured_hyperparameter_space(cls) -> dict[str, Any]:
            return structured_space_for_drp_model(cls)

        @property
        def cell_line_views(self) -> list[str]:
            return list(self._cell_line_views_list)

        @cell_line_views.setter
        def cell_line_views(self, views: list[str]) -> None:
            self._cell_line_views_list = list(views)

        @property
        def drug_views(self) -> list[str]:
            return list(self._drug_views_list)

        @drug_views.setter
        def drug_views(self, views: list[str]) -> None:
            self._drug_views_list = list(views)

        def _resolved_config(self, hyperparameters: dict[str, Any] | None = None) -> ModelConfig:
            config = config_from_build_hyperparameters(type(self), hyperparameters)
            return config or self._base_config.model_copy(deep=True)

        def build_model(self, hyperparameters: dict[str, Any]) -> None:
            self.log_hyperparameters(hyperparameters)
            self.hyperparameters = dict(hyperparameters)
            config = self._resolved_config(hyperparameters)
            self._cell_line_views_list = cell_line_views_from_model_config(config)
            self._drug_views_list = drug_views_from_model_config(config)
            self._bridge.set_composed_config(config)

        def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
            config = self._resolved_config(self.hyperparameters or None)
            views = cell_line_views_from_model_config(config)
            return load_cell_line_feature_views(
                views,
                data_path,
                dataset_name,
                model_name=self.get_model_name(),
            )

        def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
            config = self._resolved_config(self.hyperparameters or None)
            views = drug_views_from_model_config(config)
            return load_drug_feature_views(
                views,
                data_path,
                dataset_name,
                model_name=self.get_model_name(),
            )

        def train(
            self,
            output: DrugResponseDataset,
            cell_line_input: FeatureDataset,
            drug_input: FeatureDataset | None = None,
            output_earlystopping: DrugResponseDataset | None = None,
            model_checkpoint_dir: str = "checkpoints",
        ) -> None:
            _ = model_checkpoint_dir
            if self._bridge.composed is None:
                self.build_model(self.hyperparameters or self.get_default_hyperparameters())
            self._bridge.train(
                output,
                cell_line_input,
                drug_input,
                output_earlystopping=output_earlystopping,
            )

        def predict(
            self,
            cell_line_ids,
            drug_ids,
            cell_line_input: FeatureDataset,
            drug_input: FeatureDataset | None = None,
        ):
            if not self._bridge.is_trained():
                return np.full(len(cell_line_ids), np.nan)
            return self._bridge.predict(cell_line_ids, drug_ids, cell_line_input, drug_input)

        def save(self, directory: str) -> None:
            if not self._bridge.is_trained():
                msg = "Cannot save: component stack is not trained"
                raise RuntimeError(msg)
            save_component_stack(
                self._bridge,
                directory,
                hyperparameters=flat_hyperparameters_from_model_config(
                    self._resolved_config(self.hyperparameters or None),
                ),
            )

        @classmethod
        def load(cls, directory: str) -> ConstructedDRPModel:
            hyperparameters_path = os.path.join(directory, "hyperparameters.json")
            hyperparameters: dict[str, Any] = {}
            if os.path.exists(hyperparameters_path):
                with open(hyperparameters_path) as handle:
                    hyperparameters = json.load(handle)
            instance = cls()
            instance.build_model(hyperparameters)
            load_component_stack(instance._bridge, directory)
            return instance

    ConstructedDRPModel.__name__ = name
    ConstructedDRPModel.__qualname__ = name
    return ConstructedDRPModel
